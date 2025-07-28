"""
Patch-based RGB→NIR inference script
===================================
• Loads trained generator checkpoint (from rgb2nir_minimal_unet)
• Converts every RGB in --rgb_dir → NIR in --out_dir
• Handles arbitrary resolutions via tiling + overlap + smooth blending

Typical Colab:
```python
from google.colab import drive; drive.mount('/content/drive')
!python infer_rgb2nir_patch.py \
    --rgb_dir /content/drive/MyDrive/rgb_imgs \
    --ckpt  /content/drive/MyDrive/ckpt_fixed.pt \
    --out_dir /content/drive/MyDrive/nir_preds \
    --tile 512 --overlap 32
```
"""

from __future__ import annotations
import argparse
from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ------------------------------------------------------------------
#  Tiling utilities
# ------------------------------------------------------------------

def make_grid_wh(w: int, h: int, tile: int, overlap: int):
    stride = tile - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x1 = min(x + tile, w)
            y1 = min(y + tile, h)
            x0 = max(x1 - tile, 0)
            y0 = max(y1 - tile, 0)
            yield x0, y0, x1, y1


def patch_weight(tile: int, device) -> torch.Tensor:
    y = torch.linspace(-math.pi, math.pi, tile, device=device)
    w1 = 0.5 * (1 + torch.cos(y))
    w2 = w1.unsqueeze(1) @ w1.unsqueeze(0)
    # shape tile×tile → add batch & channel dims
    return w2.unsqueeze(0).unsqueeze(0)

# ------------------------------------------------------------------
#  Inference per image
# ------------------------------------------------------------------

def infer_image(net: torch.nn.Module, img: Image.Image, tile: int, overlap: int, device) -> Image.Image:
    # transforms
    tf_in = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    tf_out = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1) * 0.5),
        transforms.ToPILImage(mode='L'),
    ])

    rgb = tf_in(img).unsqueeze(0).to(device)  # 1×3×H×W
    _, _, H, W = rgb.shape

    # output & weight canvases
    out = torch.zeros((1, 1, H, W), device=device)
    wsum = torch.zeros_like(out)
    weight = patch_weight(tile, device)

    net.eval()
    with torch.no_grad():
        for x0, y0, x1, y1 in make_grid_wh(W, H, tile, overlap):
            patch = rgb[..., y0:y1, x0:x1]
            # if patch smaller than tile in border, pad
            ph, pw = y1 - y0, x1 - x0
            if ph != tile or pw != tile:
                pad = [0, tile - pw, 0, tile - ph]  # left,right,top,bottom
                patch = F.pad(patch, pad, mode='reflect')
            pred = net(patch)  # 1×1×tile×tile
            pred = pred[..., :ph, :pw]

            out[..., y0:y1, x0:x1] += pred * weight[..., :ph, :pw]
            wsum[..., y0:y1, x0:x1] += weight[..., :ph, :pw]

    avg = out / wsum.clamp(min=1e-5)
    return tf_out(avg.squeeze(0).cpu())

# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rgb_dir', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--tile', type=int, default=512)
    p.add_argument('--overlap', type=int, default=32)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = torch.device('cuda' if args.device=='cuda' and torch.cuda.is_available() else 'cpu')
    # import network
    from rgb2nir_minimal_unet import GeneratorUNet
    net = GeneratorUNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt['G'])

    rgb_dir = Path(args.rgb_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True, parents=True)

    exts = ['*.jpg','*.png','*.jpeg','*.bmp']
    files = [f for e in exts for f in sorted(rgb_dir.glob(e))]
    assert files, f'No images in {rgb_dir}'

    for img_path in files:
        img = Image.open(img_path).convert('RGB')
        nir = infer_image(net, img, args.tile, args.overlap, device)
        out_path = out_dir / img_path.with_suffix('.png').name
        nir.save(out_path)
        print(f"Saved {out_path.name}")

if __name__ == '__main__': 
    main()
