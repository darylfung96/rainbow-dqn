import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionalDQN(nn.Module):

    def __init__(self, batch_size, action_size, input_size, kernel_size: list):
        super(DistributionalDQN, self).__init__()

        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv1_output_size = input_size - kernel_size[0] + 2 + 1
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv2_output_size = self.conv1_output_size - kernel_size[1] + 2 + 1
        self.conv3 = nn.Conv2d(64, 32, 5)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.conv3_output_size = self.conv2_output_size - kernel_size[2] + 2 + 1

        self.fc1 = nn.Linear(self.conv3_output_size, 64)
        self.fc2 = nn.Linear(64, action_size) # add distribution list

        # self.policy_output = nn.Linear()

    def forward(self, x):
        conv1_output = F.relu(self.batchnorm1(self.conv1(x)))
        conv2_output = F.relu(self.batchnorm2(self.conv2(x)))
        conv3_output = F.relu(self.batchnorm3(self.conv3(x)))

        convolution_output = conv3_output.view(self.batch_size, -1)
        fc1_output = F.relu(self.fc1(convolution_output))
        fc2_output = F.relu(self.fc2(fc1_output))

