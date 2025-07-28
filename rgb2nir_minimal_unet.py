"""
Minimal paired RGB→NIR image-to-image translation prototype
===========================================================
• U-Net-style Generator (no external backbone)
• PatchGAN Discriminator
• PairedDataset that loads aligned RGB / NIR images from a single folder
  (filenames end with `_rgb.*` and `_ir.*`).
  Resizes both to given `--size` (default 512).
• Training loop with L1 + SSIM + adversarial losses

Designed for easy experimentation on Google Colab.
Python ⩾ 3.9, PyTorch ⩾ 2.2.
Optional: install `tqdm` for a progress bar.

Usage (Colab):
--------------
```bash
!pip install torch torchvision tqdm
!python rgb2nir_minimal_unet.py --data_root /content/data --size 512 --epochs 100
```
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# -------------------------------
#  Data pipeline
# -------------------------------
class PairedRGBNIR(Dataset):
    """Loads paired RGB / NIR image pairs from one folder.

    Filenames must end with `_rgb.*` for RGB and `_ir.*` for NIR.
    """
    def __init__(self, root: str | Path, size: int = 512):
        self.dir = Path(root)
        self.size = size
        self.rgb_files = sorted(self.dir.glob("*_rgb.jpg*"))
        assert self.rgb_files, f"No '*_rgb.*' files found in {self.dir}"

        self.pairs: List[Tuple[Path, Path]] = []
        for rgb_path in self.rgb_files:
            stem = rgb_path.stem.replace("_rgb", "")
            nir_glob = list(self.dir.glob(f"{stem}_ir.jpg*"))
            if not nir_glob:
                raise FileNotFoundError(f"Missing NIR for {rgb_path.name}")
            self.pairs.append((rgb_path, nir_glob[0]))

        self.tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),            # [0,1]
            transforms.Lambda(lambda x: x * 2 - 1),  # [-1,1]
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        rgb_path, nir_path = self.pairs[idx]
        rgb = self.tf(Image.open(rgb_path).convert("RGB"))
        nir = self.tf(Image.open(nir_path).convert("L"))
        return rgb, nir

# -------------------------------
#  Building blocks
# -------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, down: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=2 if down else 1, padding=1)
        self.norm = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(8, c)
        )
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x + self.block(x))

# -------------------------------
#  Generator: U-Net
# -------------------------------
class GeneratorUNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base, down=False)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)
        self.bot  = nn.Sequential(*[ResBlock(base*8) for _ in range(3)])
        self.up1  = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec1 = ConvBlock(base*8, base*4, down=False)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = ConvBlock(base*4, base*2, down=False)
        self.up3  = nn.ConvTranspose2d(base*2, base,   2, 2)
        self.dec3 = ConvBlock(base*2, base,   down=False)
        self.outc = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        d1 = self.enc1(x)
        d2 = self.enc2(d1)
        d3 = self.enc3(d2)
        d4 = self.enc4(d3)
        b  = self.bot(d4)
        u1 = torch.cat([self.up1(b), d3], 1); d5 = self.dec1(u1)
        u2 = torch.cat([self.up2(d5), d2], 1); d6 = self.dec2(u2)
        u3 = torch.cat([self.up3(d6), d1], 1); d7 = self.dec3(u3)
        return torch.tanh(self.outc(d7))

# -------------------------------
#  PatchGAN Discriminator
# -------------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 4, base: int = 64):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        nf = 1
        for i in range(1, 4): prev, nf = nf, min(2**i,8); layers += [
            nn.Conv2d(base*prev, base*nf, 4, 2, 1, bias=False),
            nn.GroupNorm(8, base*nf), nn.LeakyReLU(0.2, True)
        ]
        layers += [nn.Conv2d(base*nf,1,4,1,1)]
        self.net = nn.Sequential(*layers)
    def forward(self, rgb, nir): return self.net(torch.cat([rgb,nir],1))

# -------------------------------
#  SSIM loss
# -------------------------------
@torch.no_grad()
def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = F.avg_pool2d(img1,3,1,1); mu2 = F.avg_pool2d(img2,3,1,1)
    mu1_sq, mu2_sq = mu1*mu1, mu2*mu2; mu12 = mu1*mu2
    sigma1_sq = F.avg_pool2d(img1*img1,3,1,1)-mu1_sq
    sigma2_sq = F.avg_pool2d(img2*img2,3,1,1)-mu2_sq
    sigma12   = F.avg_pool2d(img1*img2,3,1,1)-mu12
    s = ((2*mu12+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return s.mean()

# -------------------------------
#  Loss & training step
# -------------------------------
def adv_loss(pred, real: bool): return F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred) if real else torch.zeros_like(pred))

def train_epoch(G, D, loader, og, od, dev):
    G.train(); D.train(); g_l = d_l = 0.0
    for rgb, nir in loader:
        rgb, nir = rgb.to(dev), nir.to(dev)
        od.zero_grad()
        with torch.no_grad(): fake = G(rgb)
        dr = adv_loss(D(rgb, nir), True); df = adv_loss(D(rgb, fake), False)
        ld = 0.5*(dr + df); ld.backward(); od.step()
        og.zero_grad(); fake = G(rgb)
        lg = adv_loss(D(rgb, fake), True) + F.l1_loss(fake, nir) + (1 - ssim(fake, nir))
        lg.backward(); og.step()
        g_l += lg.item(); d_l += ld.item()
    return g_l/len(loader), d_l/len(loader)

# -------------------------------
#  CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--size', type=int, default=512, help='resolution for each side')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to load/save model checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = PairedRGBNIR(args.data_root, size=args.size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, pin_memory=True, num_workers=2)

    G = GeneratorUNet().to(device)
    D = PatchDiscriminator().to(device)
    og = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for e in range(1, args.epochs + 1):
        gl, dl = train_epoch(G, D, loader, og, od, device)
        print(f"Epoch {e:03d} | G={gl:.3f} D={dl:.3f}")
        if e % 10 == 0:
            torch.save({'G': G.state_dict(), 'D': D.state_dict()}, f'ckpt_{e:03d}.pt')

if __name__ == '__main__':
    main()
