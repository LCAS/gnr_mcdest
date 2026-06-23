import torch
from torch import nn
import math


class MAM(nn.Module):
    """
    The Multi-scale Aware Module
    """
    
    def __init__(self, channels: int) -> None:
        """
        channels: The input and output depth
        """
        super().__init__()

        self.channels     = channels

        # Figure out how many channels each row of the dilated convolution layer needs
        self.outchans_1   = math.floor(self.channels / 3) + (1 if (self.channels % 3 != 0) else 0)
        self.outchans_2   = math.floor(self.channels / 3) + (1 if (self.channels % 3 == 2) else 0)
        self.outchans_3   = math.floor(self.channels / 3)

        # We need ReLU
        self.relu         = nn.ReLU()
        
        
        # The skip conv/layer
        self.c0_conv      = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding="same")


        self.row1         = nn.Sequential(
            # Regular conv
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding="same"),
            nn.BatchNorm2d(num_features=self.channels),
            nn.ReLU(),
            # Dilated conv
            nn.Conv2d(in_channels=self.channels, out_channels=self.outchans_1, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(num_features=self.outchans_1),
            nn.ReLU()
        )

        self.row2         = nn.Sequential(
            # Regular conv
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=self.channels),
            nn.ReLU(),
            # Dilated conv
            nn.Conv2d(in_channels=self.channels, out_channels=self.outchans_2, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(num_features=self.outchans_2),
            nn.ReLU()
        )

        self.row3         = nn.Sequential(
            # Regular conv
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=5, padding="same"),
            nn.BatchNorm2d(num_features=self.channels),
            nn.ReLU(),
            # Dilated conv
            nn.Conv2d(in_channels=self.channels, out_channels=self.outchans_3, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm2d(num_features=self.outchans_3),
            nn.ReLU()
        )

    def forward(self, x):
        
        # skip bit
        xr0 = self.c0_conv(x)

        # The dilated bits
        xr1 = self.row1(x)
        xr2 = self.row2(x)
        xr3 = self.row3(x)

        # Second later (concatinate of m1-3)
        # We perform the cat on dimension 1, as ∑m1-3_ochans = inchans
        conc = torch.concatenate((xr1, xr2, xr3), dim=1)
        
        # Third layer (elementwise add second layer cat and the m0conv)
        return torch.add(xr0, conc)
