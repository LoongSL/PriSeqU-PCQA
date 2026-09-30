import os
import argparse
import torch
import clip
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    sample_dirs = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ])

    extracted, skipped = 0, 0
    for name in tqdm(sample_dirs, desc="Extracting CLIP features"):
        stitched_path = os.path.join(args.data_dir, name, "stitched.png")
        save_path = os.path.join(args.data_dir, name, "clip_feat.pt")

        if not os.path.exists(stitched_path):
            skipped += 1
            continue

        img = preprocess(Image.open(stitched_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        torch.save(feat.squeeze(0).cpu(), save_path)
        extracted += 1

    print(f"Done. Extracted: {extracted}, Skipped (no stitched.png): {skipped}")


if __name__ == '__main__':
    main()
