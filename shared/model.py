import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, k_main=(1, 3, 3), mix_spectral=False):
        super().__init__()
        k2 = (3, 3, 3) if mix_spectral else k_main
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=k_main, padding=tuple(k // 2 for k in k_main), bias=False)
        self.n1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=k2, padding=tuple(k // 2 for k in k2), bias=False)
        self.n2 = nn.InstanceNorm3d(out_ch, affine=True)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )

    def forward(self, x):
        shortcut = x if self.proj is None else self.proj(x)
        x = F.relu(self.n1(self.conv1(x)))
        x = self.n2(self.conv2(x))
        x = F.relu(x + shortcut)
        return x


class ConvAE3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.x1a = ResBlock3D(1, 32, k_main=(1, 3, 3), mix_spectral=False)
        self.x1b = ResBlock3D(32, 32, k_main=(1, 3, 3), mix_spectral=False)
        self.p1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        self.x2a = ResBlock3D(32, 64, k_main=(1, 3, 3), mix_spectral=False)
        self.x2b = ResBlock3D(64, 64, k_main=(1, 3, 3), mix_spectral=True)
        self.p2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.x3a = ResBlock3D(64, 128, k_main=(1, 3, 3), mix_spectral=True)
        self.x3b = ResBlock3D(128, 128, k_main=(1, 3, 3), mix_spectral=True)
        self.p3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.b1 = ResBlock3D(128, 256, k_main=(1, 3, 3), mix_spectral=True)
        self.drop = nn.Dropout3d(p=0.05)

        self.u3 = nn.ConvTranspose3d(256, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.d3 = ResBlock3D(128 + 128, 128, k_main=(1, 3, 3), mix_spectral=True)

        self.u2 = nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.d2 = ResBlock3D(64 + 64, 64, k_main=(1, 3, 3), mix_spectral=True)

        self.u1 = nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.d1 = ResBlock3D(32 + 32, 32, k_main=(1, 3, 3), mix_spectral=False)

        self.out = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.x1b(self.x1a(x))
        p1 = self.p1(x1)

        x2 = self.x2b(self.x2a(p1))
        p2 = self.p2(x2)

        x3 = self.x3b(self.x3a(p2))
        p3 = self.p3(x3)

        b = self.drop(self.b1(p3))

        u3 = self.u3(b)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.d3(u3)

        u2 = self.u2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.d2(u2)

        u1 = self.u1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.d1(u1)

        return self.out(u1)


def sam_loss(xhat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dot = (xhat * x).sum(dim=2)
    xhat_norm = torch.sqrt((xhat * xhat).sum(dim=2) + eps)
    x_norm = torch.sqrt((x * x).sum(dim=2) + eps)
    cos = dot / (xhat_norm * x_norm + eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    ang = torch.acos(cos)
    return ang.mean()


def mse_loss(xhat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(xhat, x, reduction="mean")


def composite_loss(xhat: torch.Tensor, x: torch.Tensor, lambda_sam: float = 0.01):
    mse = mse_loss(xhat, x)
    sam = sam_loss(xhat, x)
    loss = mse + (lambda_sam * sam)
    return loss, mse.detach(), sam.detach()
