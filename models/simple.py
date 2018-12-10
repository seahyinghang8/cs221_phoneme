import torch
import torch.nn as nn
import logging


class linearRegressor(nn.Module):
    def __init__(self, num_channels, num_frames, num_classes, num_dimensions):
        # initializes the model and its weights
        # pass the config in as required
        super(linearRegressor, self).__init__()

        # we are going to regress on the middle frame
        self.frame_index = int(num_frames / 2) - 1

        self.fc = nn.Linear(num_channels * num_dimensions, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x[:, :, self.frame_index]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)

        return x