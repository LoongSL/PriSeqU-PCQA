

import torch
import torch.nn as nn
from models.backbones import resnet50
from models.xLSTM import xLSTM


class ImageRegression(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.linear1 = nn.Linear(img_dim, img_dim // 2)
        self.bn1 = nn.BatchNorm1d(img_dim // 2)
        self.linear2 = nn.Linear(img_dim // 2, img_dim // 4)
        self.bn2 = nn.BatchNorm1d(img_dim // 4)
        self.output_layer = nn.Linear(img_dim // 4, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.linear1(x)))
        x = torch.relu(self.bn2(self.linear2(x)))
        out = self.output_layer(x)
        return out[:, 0], out[:, 1]


class DeviationModule(nn.Module):
    def __init__(self, shared_dim=1024, num_heads=8):
        super().__init__()
        assert shared_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = shared_dim // num_heads
        self.scale     = self.head_dim ** -0.5


        self.to_q = nn.Linear(shared_dim, shared_dim, bias=False)

        self.to_k = nn.Linear(shared_dim, shared_dim, bias=False)
        self.to_v = nn.Linear(shared_dim, shared_dim, bias=False)

        self.to_out = nn.Linear(shared_dim, shared_dim, bias=False)
        self.norm   = nn.LayerNorm(shared_dim)

    def forward(self, visual_feat, anchor_feat):
        B, N, D = visual_feat.shape
        H, Hd   = self.num_heads, self.head_dim


        q = self.to_q(anchor_feat).view(B, N, H, Hd)   # (B, N, H, Hd)
  
        k = self.to_k(visual_feat).view(B, N, H, Hd)   # (B, N, H, Hd)
        v = self.to_v(visual_feat).view(B, N, H, Hd)   # (B, N, H, Hd)

  

        attn      = torch.softmax(
            torch.einsum('bnhd,bnkd->bnhk', q, k) * self.scale, dim=-1
        )
        retrieved = torch.einsum('bnhk,bnkd->bnhd', attn, v)  # (B, N, H, Hd)

   
        deviation = (q - retrieved).reshape(B, N, D)           # (B, N, D)

        return self.norm(self.to_out(deviation))


class SemanticPriorInitializer(nn.Module):
    def __init__(self, clip_dim=512, hidden_size=4096, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.h_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(clip_dim, hidden_size), nn.Tanh())
            for _ in range(num_layers)
        ])
        self.c_projs = nn.ModuleList([
            nn.Linear(clip_dim, hidden_size)
            for _ in range(num_layers)
        ])

    def forward(self, clip_feat):
        h_list, c_list = [], []
        for i in range(self.num_layers):
            h_list.append(self.h_projs[i](clip_feat))
            c_list.append(self.c_projs[i](clip_feat))

        h = torch.stack(h_list, dim=0)
        c = torch.stack(c_list, dim=0)
        n = torch.zeros_like(h)
        m = torch.zeros_like(h)

        return (h, c, n, m)


class res_clip_lstm_prior(nn.Module):
    def __init__(self, adjacency, clip_dropout=0.0):
        super().__init__()
        self.img_dim        = 2048
        self.clip_dim       = 512
        self.shared_dim     = 1024
        self.hidden_size    = 1024 * 4
        self.num_heads      = 4
        self.num_layers     = 2
        self.num_projections = 8
        self.clip_dropout   = clip_dropout
        self.adjacency      = adjacency

        N = self.num_projections
        adj_mask = torch.zeros(N, N)
        for i in range(N):
            for nb in adjacency[str(i)]:
                adj_mask[i, nb] = 1.0
        self.register_buffer('adj_mask', adj_mask)

        self.img_backbone = resnet50(pretrained=True)

        self.resnet_proj = nn.Sequential(
            nn.Linear(self.img_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.GELU(),
        )
        self.clip_proj = nn.Sequential(
            nn.Linear(self.clip_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.GELU(),
        )

        self.deviation = DeviationModule(self.shared_dim, num_heads=8)


        self.stream_fusion = nn.Sequential(
            nn.Linear(self.img_dim + self.shared_dim, self.img_dim),
            nn.LayerNorm(self.img_dim),
            nn.GELU(),
        )

        self.prior_init = SemanticPriorInitializer(
            clip_dim=self.clip_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )

        self.xLSTM = xLSTM(
            input_size=self.img_dim,
            head_size=1024,
            num_heads=self.num_heads,
            layers=['s', 's'],
            batch_first=True
        )

        self.regression = ImageRegression(self.img_dim * self.num_projections)

    def graph_sort(self, sort_score, fused):
    
        B, N, D = fused.shape

        visited = torch.zeros(B, N, device=fused.device)
        visited[:, 0] = 1.0
        current_onehot = torch.zeros(B, N, device=fused.device)
        current_onehot[:, 0] = 1.0

        outputs = [fused[:, 0]]

        for step in range(1, N):
            neighbors = current_onehot @ self.adj_mask    # (B, N)
            available = neighbors * (1 - visited)         # (B, N)

  
            total = available.sum(dim=-1, keepdim=True)
            fallback = (1 - visited) * (total < 1e-6).float()
            available = available + fallback


            masked_score = sort_score + torch.log(available + 1e-8)
            idx = masked_score.detach().argmax(dim=-1)    # (B,)


            current_onehot = torch.zeros(B, N, device=fused.device)
            current_onehot.scatter_(1, idx.unsqueeze(1), 1.0)

            feat = fused[torch.arange(B, device=fused.device), idx]  # (B, D)
            outputs.append(feat)

            visited = (visited + current_onehot).clamp(max=1.0)

        return torch.stack(outputs, dim=1)

    def forward(self, img, clip_feat):
        """
       
        """
        batch_size, num_projections, C, H, W = img.shape


        img_flat     = img.view(-1, C, H, W)
        img_features = self.img_backbone(img_flat)
        img_features = torch.flatten(img_features, 1)
        img_features = img_features.view(batch_size, num_projections, self.img_dim)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

  
        clip_feat = clip_feat.to(self.resnet_proj[0].weight.dtype)
        if self.training and self.clip_dropout > 0:
            mask = (torch.rand(batch_size, 1, device=clip_feat.device) > self.clip_dropout).float()
            clip_feat = clip_feat * mask


        visual_shared  = self.resnet_proj(img_features)
        clip_expanded  = clip_feat.unsqueeze(1).expand(-1, num_projections, -1)
        anchor_shared  = self.clip_proj(clip_expanded)


        deviation_feat = self.deviation(visual_shared, anchor_shared)

  
        fused = self.stream_fusion(
            torch.cat([img_features, deviation_feat], dim=-1)
        )

       
        sort_score = deviation_feat.norm(dim=-1)                          # (B, N)

      
        fused = self.graph_sort(sort_score, fused)               # (B, N, D)

   
        prior_state = self.prior_init(clip_feat)


        lstm_out, _ = self.xLSTM(fused, state=prior_state)


        features = lstm_out.reshape(batch_size, -1)
        mean_score, log_variance = self.regression(features)
        return mean_score, log_variance
