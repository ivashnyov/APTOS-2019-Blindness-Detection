import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepEmotion(nn.Module):

    def __init__(self, num_classes=7):
        super(DeepEmotion, self).__init__()
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_localization = nn.Sequential(
            nn.Linear(in_features=10 * 59 * 59, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_localization[2].weight.data.zero_()
        self.fc_localization[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc = nn.Sequential(
            nn.Linear(in_features=10 * 244 * 244, out_features=50),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=50, out_features=self.num_classes)
        )

    def stn_forward(self, x):
        x_spatial = self.localization(x)
        x_convolved = self.layer(x)
        x_spatial = x_spatial.view(-1, 10 * 59 * 59)
        theta = self.fc_localization(x_spatial)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x_convolved, grid)
        return x

    def forward(self, x):
        x = self.stn_forward(x)
        x = x.view(-1, 10 * 244 * 244)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)