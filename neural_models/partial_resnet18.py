"""First layers of ResNet-18, max-pooled to one value per Warcraft cell: (B, k, k, 8, 8, 3) -> (B, 5, k, k)."""
from torchvision.models import resnet18
import torch
from torch import nn
import torch.nn.functional as F


class PartialResNet(nn.Module):
    """softmax(-alpha * |h - center|**beta) over 5 class centers: the cell costs or learnable (init 0..4)."""

    def __init__(self, k, alpha=2.0, beta=1.0, learnable_class_centers=False):
        super().__init__()
        resnet = resnet18(pretrained=False)
        # first layers of ResNet-18
        self.conv1   = resnet.conv1
        self.bn      = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.block   = resnet.layer1
        self.register_buffer(
            "WEIGHTS",
            torch.tensor([0.8, 1.2, 5.3, 7.7, 9.2]).view(1, 5, 1, 1)
        )
        self.learnable_class_centers = learnable_class_centers
        if learnable_class_centers:
            self.class_centers = nn.Parameter(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).view(1, 5, 1, 1))
        self.alpha = alpha
        self.beta = beta

        # one scalar per position
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=True
        )

        # pooling to one value per cell
        self.maxpool2 = nn.AdaptiveMaxPool2d((k, k))

    def forward(self, x):  # x: (B, k, k, 8, 8, 3)
        B, dim, _, gx, gy, _ = x.shape
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, 3, dim*gx, dim*gy)  # (B, 3, 8k, 8k)
        h = self.conv1(x)
        h = self.bn(h)
        h = self.relu(h)
        h = self.maxpool(h)
        h = self.block(h)
        h = self.conv2(h)  # (B, 1, H, W)
        h = self.maxpool2(h)  # (B, 1, k, k)
        centers = self.class_centers if self.learnable_class_centers else self.WEIGHTS
        dist = torch.abs(h - centers)**self.beta
        return F.softmax(-self.alpha*dist, dim=1)  # (B, 5, k, k)
