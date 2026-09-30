import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedUncertaintyLoss(nn.Module):
    def __init__(self, l2_weight=1.0, rank_weight=1.0, nll_weight=1.0, hard_thred=1.0, use_margin=False):
        super(CombinedUncertaintyLoss, self).__init__()
        self.l2_weight = l2_weight
        self.rank_weight = rank_weight
        self.nll_weight = nll_weight
        self.hard_thred = hard_thred
        self.use_margin = use_margin
        self.nll_loss = nn.GaussianNLLLoss(reduction='mean')
    def forward(self, mean_pred, log_var_pred, gts):
        mean_pred = mean_pred.view(-1)
        log_var_pred = log_var_pred.view(-1)
        gts = gts.view(-1)
        variance_pred = torch.exp(log_var_pred)
        nll_loss = self.nll_loss(mean_pred, gts, variance_pred) * self.nll_weight
        l2_loss = F.mse_loss(mean_pred, gts) * self.l2_weight
        n = len(mean_pred)
        preds = mean_pred.unsqueeze(0).repeat(n, 1)
        preds_t = preds.t()
        img_label = gts.unsqueeze(0).repeat(n, 1)
        img_label_t = img_label.t()
        masks = torch.sign(img_label - img_label_t)
        masks_hard = (torch.abs(img_label - img_label_t) < self.hard_thred) & (torch.abs(img_label - img_label_t) > 0)
        if self.use_margin:
            rank_loss = masks_hard * torch.relu(torch.abs(img_label - img_label_t) - masks * (preds - preds_t))
        else:
            rank_loss = masks_hard * torch.relu(-masks * (preds - preds_t))
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08) * self.rank_weight
        total_loss = nll_loss + l2_loss + rank_loss
        return total_loss