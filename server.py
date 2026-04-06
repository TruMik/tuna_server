import io
import os
import torch
import torch.nn as nn
import numpy as np
import gdown
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights


# ── Auto-download model from Google Drive ────────────────────────────────────

MODEL_PATH = "mlr_vggnet_optuna_best.pth"
GDRIVE_ID  = "1kaUgo6XLK5Q5WIX8oGdVUHYzcI3h2uFA"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[...] Downloading model from Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_ID}",
            MODEL_PATH,
            quiet=False
        )
        print("[OK] Model downloaded")
    else:
        print("[OK] Model already exists")

download_model()


# ── Model Definition ──────────────────────────────────────────────────────────

class AsymmetricConvBN(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.conv_31 = nn.Conv2d(channels, channels, kernel_size=(3, 1),
                                 padding=(1, 0), bias=False)
        self.bn_31   = nn.BatchNorm2d(channels)
        self.conv_13 = nn.Conv2d(channels, channels, kernel_size=(1, 3),
                                 padding=(0, 1), bias=False)
        self.bn_13   = nn.BatchNorm2d(channels)
        self.relu    = nn.ReLU(inplace=False)
        self.drop    = nn.Dropout2d(dropout)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn_31(self.conv_31(x)))
        out = self.bn_13(self.conv_13(out))
        out = self.drop(out)
        out = self.relu(out + identity)
        return out


class DSCProjection(nn.Module):
    def __init__(self, in_ch, out_ch, pool_stride):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3,
                                   padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn        = nn.BatchNorm2d(out_ch)
        self.relu      = nn.ReLU(inplace=False)
        self.pool      = nn.MaxPool2d(kernel_size=pool_stride,
                                      stride=pool_stride)

    def forward(self, x):
        x = self.relu(self.bn(self.pointwise(self.depthwise(x))))
        x = self.pool(x)
        return x


class MLRVGGNet(nn.Module):
    def __init__(self, num_classes=3, dropout=0.0, ac_dropout=0.0):
        super().__init__()
        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.block1 = feats[0:5]
        self.block2 = feats[5:10]
        self.block3 = feats[10:17]
        self.block4 = feats[17:24]
        for block in [self.block1, self.block2,
                      self.block3, self.block4]:
            for param in block.parameters():
                param.requires_grad = False
        self.mlr_proj1 = DSCProjection(64,  512, pool_stride=8)
        self.mlr_proj2 = DSCProjection(128, 512, pool_stride=4)
        self.mlr_proj3 = DSCProjection(256, 512, pool_stride=2)
        self.ac1 = AsymmetricConvBN(512, dropout=ac_dropout)
        self.ac2 = AsymmetricConvBN(512, dropout=ac_dropout)
        self.ac3 = AsymmetricConvBN(512, dropout=ac_dropout)
        self.fusion_bn  = nn.BatchNorm2d(512)
        self.final_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.head       = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        p1 = self.mlr_proj1(b1)
        p2 = self.mlr_proj2(b2)
        p3 = self.mlr_proj3(b3)
        fused = self.fusion_bn(b4 + p1 + p2 + p3)
        out = self.ac1(fused)
        out = self.ac2(out)
        out = self.ac3(out)
        out = self.final_pool(out)
        out = self.gap(out).flatten(1)
        return self.head(out)


# ── App Setup ─────────────────────────────────────────────────────────────────

CLASSES = ["Mackerel Tuna", "Skipjack Tuna", "Yellowfin Tuna"]
DEVICE  = torch.device("cpu")  # Render free tier has no GPU

model = MLRVGGNet(num_classes=3, dropout=0.0, ac_dropout=0.0).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"[OK] Model ready on {DEVICE}")

transform = transforms.Compose([
    transforms.Resize(int(224 * 1.14)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

app = FastAPI(title="Tuna Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status":  "ok",
        "model":   "MLR-VGGNet",
        "classes": CLASSES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img      = Image.open(io.BytesIO(contents)).convert("RGB")
    x        = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(np.argmax(probs))

    return {
        "class":         CLASSES[pred_idx],
        "confidence":    float(probs[pred_idx]),
        "probabilities": {
            c: float(p) for c, p in zip(CLASSES, probs)
        },
    }
