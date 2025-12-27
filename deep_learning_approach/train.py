import os
import cv2
import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# =========================
# DATASET
# =========================
class TuSimpleLaneDataset(Dataset):
    def __init__(self, root, split="training"):
        root = Path(root) / split
        frame_dir = root / "frames"
        mask_dir  = root / "lane-masks"

        frames = sorted(frame_dir.glob("*.jpg"))
        masks  = sorted(mask_dir.glob("*.jpg"))   # MASKLER DE JPG

        frame_dict = {p.stem: p for p in frames}
        mask_dict  = {p.stem: p for p in masks}

        common = sorted(list(set(frame_dict.keys()) & set(mask_dict.keys())))
        self.samples = [(frame_dict[k], mask_dict[k]) for k in common]

        print("Total frames:", len(frames))
        print("Total masks :", len(masks))
        print("USABLE PAIRS:", len(self.samples))

        assert len(self.samples) > 0, "No usable pairs found!"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # ---- Image ----
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 256))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        # ---- Mask ----
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 256), interpolation=cv2.INTER_NEAREST)

        # 🔥 ÖNEMLİ: Binary lane mask (0 = background, 1 = lane)
        mask = (mask > 0).astype(np.int64)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


# =========================
# U-NET
# =========================
class DoubleConv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(True),
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2):  # 🔥 num_classes = 2 (lane / no-lane)
        super().__init__()

        self.d1 = DoubleConv(in_ch, 64);  self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128);    self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(128, 256);   self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(256, 512);   self.p4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.u4 = nn.ConvTranspose2d(1024, 512, 2, 2); self.c4 = DoubleConv(1024, 512)
        self.u3 = nn.ConvTranspose2d(512, 256, 2, 2);  self.c3 = DoubleConv(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2);  self.c2 = DoubleConv(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2);   self.c1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1 = self.d1(x); x2 = self.p1(x1)
        x3 = self.d2(x2); x4 = self.p2(x3)
        x5 = self.d3(x4); x6 = self.p3(x5)
        x7 = self.d4(x6); x8 = self.p4(x7)

        b = self.bottleneck(x8)

        x = self.u4(b); x = self.c4(torch.cat([x, x7], 1))
        x = self.u3(x); x = self.c3(torch.cat([x, x5], 1))
        x = self.u2(x); x = self.c2(torch.cat([x, x3], 1))
        x = self.u1(x); x = self.c1(torch.cat([x, x1], 1))

        return self.out(x)


# =========================
# TRAIN
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE =", DEVICE)

dataset = TuSimpleLaneDataset("/kaggle/input/tusimple")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet(num_classes=2).to(DEVICE)

# Binary olduğu için normal CrossEntropyLoss yeterli
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# İlk batch'ten mask değerlerine bak (debug için)
imgs_dbg, masks_dbg = next(iter(loader))
print("DEBUG mask unique values:", torch.unique(masks_dbg))

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0

    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)          # [B, 2, H, W]
        loss = criterion(logits, masks)  # masks ∈ {0,1}
        loss.backward()
        optimizer.step()

        running += loss.item()

    print(f"Epoch {epoch} Loss = {running/len(loader):.4f}")

torch.save(model.state_dict(), "/kaggle/working/tusimple_unet_binary.pth")
print("MODEL SAVED ✔️  (/kaggle/working/tusimple_unet_binary.pth)")