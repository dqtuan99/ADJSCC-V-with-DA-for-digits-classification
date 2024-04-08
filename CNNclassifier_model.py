# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:11:49 2024

@author: Tuan
"""

import torch.nn as nn
import torch.nn.functional as F

class DigitClassifierCNN(nn.Module):
    def __init__(self):
        super(DigitClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout1 = nn.Dropout2d(0.25)  # Dropout for conv layers
        self.dropout2 = nn.Dropout(0.5)  # Dropout for fully connected layers


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 32x32 -> 16x16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 16x16 -> 8x8
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # 8x8 -> 4x4
        x = self.dropout1(x)
        x = x.view(-1, 128 * 4 * 4)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))

        return F.log_softmax(self.fc3(x), dim=1)