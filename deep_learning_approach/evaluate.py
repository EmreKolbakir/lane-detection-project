import cv2
import torch
import numpy as np
from torch import nn

# ================= MODEL =================
class DoubleConv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(a,b,3,padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(True),
            nn.Conv2d(b,b,3,padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(True)
        )
    def forward(self,x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.d1 = DoubleConv(in_ch,64);  self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64,128);    self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(128,256);   self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(256,512);   self.p4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512,1024)

        self.u4 = nn.ConvTranspose2d(1024,512,2,2); self.c4 = DoubleConv(1024,512)
        self.u3 = nn.ConvTranspose2d(512,256,2,2);  self.c3 = DoubleConv(512,256)
        self.u2 = nn.ConvTranspose2d(256,128,2,2);  self.c2 = DoubleConv(256,128)
        self.u1 = nn.ConvTranspose2d(128,64,2,2);   self.c1 = DoubleConv(128,64)

        self.out = nn.Conv2d(64,num_classes,1)

    def forward(self,x):
        x1=self.d1(x); x2=self.p1(x1)
        x3=self.d2(x2); x4=self.p2(x3)
        x5=self.d3(x4); x6=self.p3(x5)
        x7=self.d4(x6); x8=self.p4(x7)

        b=self.bottleneck(x8)

        x=self.u4(b); x=self.c4(torch.cat([x,x7],1))
        x=self.u3(x); x=self.c3(torch.cat([x,x5],1))
        x=self.u2(x); x=self.c2(torch.cat([x,x3],1))
        x=self.u1(x); x=self.c1(torch.cat([x,x1],1))

        return self.out(x)

# ================= DEVICE =================
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("DEVICE =", DEVICE)

MODEL_PATH = "model/tusimple_unet_binary.pth"
VIDEO_PATH = "data/videos/normalDay/nD_16.mp4"
OUTPUT_PATH = "ego_lane_video.mp4"

# ================= LOAD MODEL =================
model = UNet(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ================= VIDEO =================
cap = cv2.VideoCapture(VIDEO_PATH)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w,h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()

    # -------- INFERENCE --------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb,(512,256))
    rgb = rgb.astype(np.float32)/255.0
    rgb = rgb.transpose(2,0,1)
    rgb = torch.from_numpy(rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out_logits = model(rgb)
        pred = torch.argmax(out_logits, dim=1).cpu().numpy()[0]

    mask = (pred==1).astype(np.uint8)*255
    mask = cv2.resize(mask,(w,h))

    # -------- CONTOURS --------
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx = w // 2
    left_cands = []
    right_cands = []

    for c in cnts:
        if cv2.contourArea(c) < 500:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        x = int(M["m10"] / M["m00"])

        if x < cx:
            left_cands.append((cx-x,c))
        else:
            right_cands.append((x-cx,c))

    left_cands.sort(key=lambda x: x[0])
    right_cands.sort(key=lambda x: x[0])

    road = np.zeros_like(mask)

    if len(left_cands)>0 and len(right_cands)>0:
        left = left_cands[0][1][:,0,:]
        right = right_cands[0][1][:,0,:]

        left  = left[np.argsort(left[:,1])]
        right = right[np.argsort(right[:,1])]

        left  = left[::2]
        right = right[::2]

        polygon = np.vstack([left, right[::-1]])

        cv2.fillPoly(road,[polygon.astype(np.int32)],255)

        kernel = np.ones((15,15),np.uint8)
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel)

    else:
        road = mask

    # -------- DRAW --------
    color = np.zeros_like(orig)
    color[:,:,1] = road
    overlay = cv2.addWeighted(orig,1.0,color,0.6,0)

    out.write(overlay)

    cv2.imshow("Lane", overlay)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("DONE! Saved:", OUTPUT_PATH)