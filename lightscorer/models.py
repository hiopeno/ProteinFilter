from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 轻量增强模块：大核上下文 + GRN（ConvNeXt V2 / RepLKNet 思路）
# ---------------------------------------------------------------------------


class GRN(nn.Module):
    """Global Response Normalization，增强通道间竞争而不显著增重。"""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


class LargeKernelContextBlock(nn.Module):
    """用 depthwise 大核补充中远程依赖，同时保持残差路径稳定。"""

    def __init__(self, channels: int, kernel_size: int, use_grn: bool = True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding")
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.grn = GRN(channels) if use_grn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = self.grn(x)
        return x + residual


# ---------------------------------------------------------------------------
# RepVGG Block: 训练时 3×3 + 1×1 + Identity，推理时可融合为单一 3×3
# ---------------------------------------------------------------------------


def _conv_bn(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
    )


class RepVGGBlock(nn.Module):
    """RepVGG 风格块：训练时三分支，推理时可融合为单 3×3 卷积。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = 1

        self.nonlinearity = nn.ReLU(inplace=True)

        # 仅当 in_ch==out_ch 且 stride==1 时使用 identity
        self.rbr_identity = (
            nn.BatchNorm2d(num_features=in_channels)
            if (out_channels == in_channels and stride == 1)
            else None
        )
        self.rbr_dense = _conv_bn(in_channels, out_channels, 3, stride=stride, padding=1)
        padding_1x1 = 1 - 3 // 2  # 0
        self.rbr_1x1 = _conv_bn(in_channels, out_channels, 1, stride=stride, padding=padding_1x1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.forward_deploy(x)
        if self.rbr_identity is None:
            id_out = 0.0
        else:
            id_out = self.rbr_identity(x)
        return self.nonlinearity(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    def _fuse_bn_tensor(self, branch) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if branch is None:
            return None, None
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            bn = branch[1]
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            kernel = None
            bn = branch
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        if kernel is not None:
            kernel_fused = kernel * t
        else:
            # Identity: 3×3 中心为 1 的核
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            kernel_value = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel_fused = kernel_value * t
        bias_fused = beta - running_mean * gamma / std
        return kernel_fused, bias_fused

    def _pad_1x1_to_3x3(self, kernel: torch.Tensor) -> torch.Tensor:
        if kernel is None:
            return None
        return torch.nn.functional.pad(kernel, [1, 1, 1, 1])

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        k3, b3 = self._fuse_bn_tensor(self.rbr_dense)
        k1, b1 = self._fuse_bn_tensor(self.rbr_1x1)
        kid, bid = self._fuse_bn_tensor(self.rbr_identity)

        kernel = k3.clone()
        if k1 is not None:
            kernel = kernel + self._pad_1x1_to_3x3(k1)
        if kid is not None:
            kernel = kernel + kid  # identity 已是 3×3
        bias = b3 + b1 + (bid if bid is not None else 0.0)
        return kernel, bias

    def switch_to_deploy(self) -> None:
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        del self.rbr_dense
        del self.rbr_1x1
        if self.rbr_identity is not None:
            del self.rbr_identity
        self.rbr_identity = None

    def forward_deploy(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinearity(self.rbr_reparam(x))


# ---------------------------------------------------------------------------
# ShiftwiseConv Block: 3×3 小核 + 空间移位模拟大感受野（CVPR 2025）
# ---------------------------------------------------------------------------


def _shift_2d(x: torch.Tensor, shift_h: int, shift_w: int) -> torch.Tensor:
    """沿 H、W 维度做循环移位。shift_h>0 向下，shift_w>0 向右。"""
    if shift_h == 0 and shift_w == 0:
        return x
    return torch.roll(x, shifts=(shift_h, shift_w), dims=(2, 3))


class ShiftwiseConvBlock(nn.Module):
    """ShiftwiseConv 风格块：四方向空间移位 + 3×3 卷积，用 3×3 小核模拟大感受野。
    参考：ShiftwiseConv (CVPR 2025) - Small Convolutional Kernel with Large Kernel Effect。
    机制：多路径长距离稀疏依赖，通过空间移位增强特征利用，硬件友好。
    """

    def __init__(self, in_ch: int, out_ch: int, shift_stride: int = 1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.shift_stride = shift_stride
        self.n_groups = 4
        # 不足 4 通道时复制，否则均分 4 组
        self.group_ch = in_ch if in_ch < self.n_groups else in_ch // self.n_groups
        self.shifted_ch = self.group_ch * self.n_groups
        self.conv = nn.Conv2d(self.shifted_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道分组：不足 4 通道时复制；否则均分 4 组
        if self.in_ch < self.n_groups:
            parts = [x] * self.n_groups  # 每份 (N, in_ch, H, W)，concat 后 (N, in_ch*4, H, W)
        else:
            ch_per = self.in_ch // self.n_groups
            parts = [
                x[:, 0:ch_per],
                x[:, ch_per : 2 * ch_per],
                x[:, 2 * ch_per : 3 * ch_per],
                x[:, 3 * ch_per : 4 * ch_per],
            ]
        s = self.shift_stride
        shifted = [
            _shift_2d(parts[0], -s, 0),   # 上
            _shift_2d(parts[1], s, 0),    # 下
            _shift_2d(parts[2], 0, -s),   # 左
            _shift_2d(parts[3], 0, s),    # 右
        ]
        y = torch.cat(shifted, dim=1)
        return self.bn(self.conv(y))


# ---------------------------------------------------------------------------
# PConv Block: 仅对 1/4 通道做空间卷积，降低内存访问
# ---------------------------------------------------------------------------


class PConvBlock(nn.Module):
    """部分卷积块：仅对前 r 比例通道做 3×3 空间卷积，其余穿透。"""

    def __init__(self, in_ch: int, out_ch: int, r: float = 0.25):
        super().__init__()
        self.part_ch = max(1, int(in_ch * r))
        self.spatial_conv = nn.Conv2d(self.part_ch, self.part_ch, 3, padding=1)
        self.proj = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, : self.part_ch], x[:, self.part_ch :]
        y1 = self.spatial_conv(x1)
        y = torch.cat([y1, x2], dim=1)
        return self.proj(y)


# ---------------------------------------------------------------------------
# 基础 CNN 模型
# ---------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x).flatten(1)
        return self.head(feat).squeeze(-1)


class ImprovedCNN(nn.Module):
    """加深加宽 + BatchNorm + Dropout 的优化版 CNN。"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.head(feat).squeeze(-1)


class ImprovedCNN_GRN(nn.Module):
    """在 ImprovedCNN 的中后层加入 GRN，增强通道竞争但不改感受野。"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            GRN(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            GRN(256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.head(feat).squeeze(-1)


class ImprovedCNN_LargeKernel(nn.Module):
    """在 ImprovedCNN 的中后层加入大核上下文块，但不使用 GRN。"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            LargeKernelContextBlock(128, kernel_size=7, use_grn=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            LargeKernelContextBlock(256, kernel_size=9, use_grn=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.head(feat).squeeze(-1)


class ImprovedCNN_LK_GRN(nn.Module):
    """以 ImprovedCNN 为骨架，在中后层加入大核上下文与 GRN。"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            LargeKernelContextBlock(128, kernel_size=7),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            LargeKernelContextBlock(256, kernel_size=9),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.head(feat).squeeze(-1)


# ---------------------------------------------------------------------------
# 消融实验变体：RepVGG、PConv、RepVGG+PConv
# ---------------------------------------------------------------------------


class ImprovedCNN_RepVGG(nn.Module):
    """ImprovedCNN + RepVGG：每个卷积块替换为 RepVGGBlock。"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            RepVGGBlock(in_channels, 32, stride=1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            RepVGGBlock(32, 64, stride=1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            RepVGGBlock(64, 128, stride=1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            RepVGGBlock(128, 256, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.head(feat).squeeze(-1)

    def fuse_for_inference(self) -> None:
        """融合所有 RepVGGBlock 为单路 3×3，用于推理加速。"""
        for m in self.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()


class ImprovedCNN_PConv(nn.Module):
    """ImprovedCNN + PConv：每个卷积块替换为 PConvBlock + BN。"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            PConvBlock(in_channels, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            PConvBlock(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            PConvBlock(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            PConvBlock(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.head(feat).squeeze(-1)


class ImprovedCNN_PConv_05(nn.Module):
    """ImprovedCNN + PConv(r=0.5)：半数通道做空间卷积，其余与 PConv 一致。"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            PConvBlock(in_channels, 32, r=0.5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            PConvBlock(32, 64, r=0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            PConvBlock(64, 128, r=0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            PConvBlock(128, 256, r=0.5),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.head(feat).squeeze(-1)


class ImprovedCNN_ShiftwiseConv(nn.Module):
    """ImprovedCNN + ShiftwiseConv：每个卷积块替换为 ShiftwiseConvBlock。
    3×3 小核 + 四方向空间移位，模拟大感受野，硬件友好。
    """

    def __init__(self, in_channels: int = 1, shift_stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            ShiftwiseConvBlock(in_channels, 32, shift_stride=shift_stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            ShiftwiseConvBlock(32, 64, shift_stride=shift_stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            ShiftwiseConvBlock(64, 128, shift_stride=shift_stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            ShiftwiseConvBlock(128, 256, shift_stride=shift_stride),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.head(feat).squeeze(-1)


class ImprovedCNN_ShiftwiseConv_S2(nn.Module):
    """ImprovedCNN + ShiftwiseConv(shift_stride=2)：空间移位步长 2，等效感受野更大。
    用于消融实验，对比 shift_stride=1 与 2 对 recall 与泛化的影响。
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            ShiftwiseConvBlock(in_channels, 32, shift_stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            ShiftwiseConvBlock(32, 64, shift_stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            ShiftwiseConvBlock(64, 128, shift_stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            ShiftwiseConvBlock(128, 256, shift_stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.head(feat).squeeze(-1)


class ImprovedCNN_RepVGG_PConv(nn.Module):
    """ImprovedCNN + RepVGG + PConv：每个 Block 为 PConvBlock → RepVGGBlock（方案 A 串联）。"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            PConvBlock(in_channels, 32),
            RepVGGBlock(32, 32, stride=1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            PConvBlock(32, 64),
            RepVGGBlock(64, 64, stride=1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            PConvBlock(64, 128),
            RepVGGBlock(128, 128, stride=1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            PConvBlock(128, 256),
            RepVGGBlock(256, 256, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.head(feat).squeeze(-1)

    def fuse_for_inference(self) -> None:
        """融合所有 RepVGGBlock 为单路 3×3。"""
        for m in self.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
