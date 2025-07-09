# Shiny app for Guided Super‑Resolution demo (GSR) – Python version
# ---------------------------------------------------------------
# Requires `shiny>=0.9`.  Uses new sidebar API.
# ---------------------------------------------------------------

import os
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.exposure import match_histograms
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from super_image import EdsrModel, ImageLoader
from shiny import App, reactive, render, ui

# ---------------------------------------------------------------------
# Global settings & deterministic seed
# ---------------------------------------------------------------------
SEED = 6740  # user‑preferred seed
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            if bilinear
            else nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class TwoInputUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        # src encoder
        self.s1 = DoubleConv(n_channels, 64)
        self.s2 = Down(64, 128)
        self.s3 = Down(128, 256)
        self.s4 = Down(256, 512)
        self.s5 = Down(512, 512)
        # ref encoder
        self.r1 = DoubleConv(n_channels, 64)
        self.r2 = Down(64, 128)
        self.r3 = Down(128, 256)
        self.r4 = Down(256, 512)
        self.r5 = Down(512, 512)
        # decoder
        self.u1 = Up(1024, 256, bilinear)
        self.u2 = Up(512, 128, bilinear)
        self.u3 = Up(256, 64, bilinear)
        self.u4 = Up(128, 64, bilinear)
        self.out = OutConv(64, n_classes)

    def forward(self, src, ref):
        s1, s2, s3, s4, s5 = self.s1(src), None, None, None, None
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        s5 = self.s5(s4)
        r1, r2, r3, r4, r5 = self.r1(ref), None, None, None, None
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        r4 = self.r4(r3)
        r5 = self.r5(r4)
        bottom = s5 + r5
        x = self.u1(bottom, s4 + r4)
        x = self.u2(x, s3 + r3)
        x = self.u3(x, s2 + r2)
        x = self.u4(x, s1 + r1)
        return self.out(x)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
EDS_MODEL = None


def edsr_upsample(img: np.ndarray) -> np.ndarray:
    """Apply x4 EDSR; return float32 0‑1 gray image."""
    global EDS_MODEL
    if EDS_MODEL is None:
        EDS_MODEL = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=4)
        EDS_MODEL.eval()
    img_u8 = ((img - img.min()) / (img.max() - img.min() + 1e-6) * 255).astype(np.uint8)
    pil = Image.fromarray(img_u8).convert("RGB")
    with torch.no_grad():
        inp = ImageLoader.load_image(pil).to(device)
        sr = EDS_MODEL(inp).cpu().squeeze().numpy()
    sr = np.transpose(sr, (1, 2, 0)).astype(np.float32)
    sr_gray = Image.fromarray((sr * 255).clip(0, 255).astype(np.uint8)).convert("L")
    return np.array(sr_gray, dtype=np.float32) / 255.0


def combined_loss(pred, src, ref, w):
    mse = nn.functional.mse_loss(pred, ref)
    ssim_loss = 1 - ssim(pred, src, data_range=1, size_average=True)
    return (1 - w) * mse + w * ssim_loss


def run_pipeline(guide_path: Path, low_path: Path, epochs: int, ssim_w: float, prog=None):
    guide = cv2.imread(str(guide_path), cv2.IMREAD_GRAYSCALE)
    low = cv2.imread(str(low_path), cv2.IMREAD_GRAYSCALE)
    guide = cv2.resize(guide, (1024, 1024), interpolation=cv2.INTER_AREA) / 255.0
    low_small = cv2.resize(low, (64, 64), interpolation=cv2.INTER_NEAREST)
    low_sr = edsr_upsample(low_small)
    low_sr = cv2.resize(low_sr, (1024, 1024), interpolation=cv2.INTER_AREA)

    g = torch.tensor(guide, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    l = torch.tensor(low_sr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    model = TwoInputUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    model.train()
    for i in range(epochs):
        opt.zero_grad()
        out = model(g, l)
        loss = combined_loss(out, g, l, ssim_w)
        loss.backward()
        opt.step()
        if prog:
            prog.set(i + 1)  # update progress bar

    model.eval()
    with torch.no_grad():
        out = torch.sigmoid(model(g, l)).cpu().squeeze().numpy()
    out = match_histograms(out, low / 255.0, channel_axis=None)
    return out


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("Guided Super‑Resolution"),
        ui.input_file("guide", "Guide (high‑res structure)"),
        ui.input_file("lowres", "Low‑res MSI"),
        ui.input_numeric("epochs", "Epochs", 50, min=1, max=500),
        ui.input_slider("ssim_w", "SSIM weight", 0.0, 1.0, 0.15, step=0.05),
        ui.input_action_button("run", "Run SR", class_="btn-primary"),
    ),
    ui.layout_columns(
        ui.output_image(id="guide_plot"),
        ui.output_image(id="low_plot"),
        ui.output_image(id="sr_plot"),
        col_widths=[4, 4, 4],
    ),
)


# ---------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------

def server(input, output, session):
    def _read_gray(fileinfo):
        if not fileinfo:
            return None
        return cv2.imread(fileinfo[0]["datapath"], cv2.IMREAD_GRAYSCALE)

    def _save_temp(img, cmap="gray") -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.imsave(tmp.name, img, cmap=cmap)
        return tmp.name

    @reactive.Calc
    def guide_img():
        return _read_gray(input.guide())

    @reactive.Calc
    def low_img():
        return _read_gray(input.lowres())

    # Raw previews
    @output
    @render.image
    def guide_plot():
        img = guide_img()
        if img is None:
            return None
        return {"src": _save_temp(img), "width": "100%"}

    @output
    @render.image
    def low_plot():
        img = low_img()
        if img is None:
            return None
        return {"src": _save_temp(img), "width": "100%"}

    # Super‑resolution output—triggered by button
    @output
    @render.image
    def sr_plot():
        input.run()  # depend on button clicks
        if not input.guide() or not input.lowres():
            return None
        with ui.Progress(min=0, max=input.epochs()) as prog:
            sr = run_pipeline(
                Path(input.guide()[0]["datapath"]),
                Path(input.lowres()[0]["datapath"]),
                int(input.epochs()),
                float(input.ssim_w()),
                prog=prog  # <- pass the progress bar handle
            )
            prog.set(input.epochs())
        return {"src": _save_temp(sr, cmap="viridis"), "width": "100%"}

# ---------------------------------------------------------------------
# Expose app for Shiny/uvicorn
# ---------------------------------------------------------------------
app = App(app_ui, server)

if __name__ == "__main__":
    import shiny
    shiny.run_app(app, reload=False)

#shiny run --reload app.py
#shiny run --reload --launch-browser app.py