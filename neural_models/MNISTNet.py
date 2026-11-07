"""LeNet-style MNIST classifier (from A-NeSI / DeepProbLog), applied to every digit."""
import torch.nn as nn


class MNIST_Net(nn.Module):
    def __init__(self, N=10, with_softmax=True, size=16 * 4 * 4):
        super(MNIST_Net, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        self.N = N
        if with_softmax:
            if N == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, N),
        )

    def forward(self, x):
        """x: (b, digits, 28, 28) -> (b, digits, N)"""
        # shape [b, ds, 1, 28, 28] is also possible - will be reshaped anyway
        batch_size = x.shape[0]
        # transform x into [b*ds, 1, 28, 28]
        x = x.reshape(-1, 1, 28, 28)
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        x = x.reshape(batch_size, -1, self.N) # shape [b, ds, N]
        return x
