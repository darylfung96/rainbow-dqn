import torch.nn as nn
import torch.nn.functional as F


class RainbowDQN(nn.Module):

    def __init__(self, action_size, atom_size, input_size, kernel_size: list):
        super(RainbowDQN, self).__init__()

        self.action_size = action_size
        self.atom_size = atom_size

        self.conv1 = nn.Conv2d(3, 32, kernel_size[0])
        self.conv1_output_size = input_size - kernel_size[0] + 0 + 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size[1])
        self.conv2_output_size = self.conv1_output_size - kernel_size[1] + 0 + 1
        self.conv3 = nn.Conv2d(64, 32, kernel_size[2])
        self.conv3_output_size = self.conv2_output_size - kernel_size[2] + 0 + 1

        # self.fc1 = nn.Linear(self.conv3_output_size ** 2 * 32, 64)
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)

        ###
        self.fc_value = nn.Linear(32, self.atom_size)
        self.fc_advantage = nn.Linear(32, self.action_size * self.atom_size)

    def forward(self, x):
        # conv1_output = F.relu(self.batchnorm1(self.conv1(x)))
        # conv2_output = F.relu(self.batchnorm2(self.conv2(conv1_output)))
        # conv3_output = F.relu(self.batchnorm3(self.conv3(conv2_output)))
        #
        # convolution_output = conv3_output.view(1, -1)
        # fc1_output = F.relu(self.fc1(convolution_output))
        fc1_output = F.relu(self.fc1(x))
        fc2_output = F.relu(self.fc2(fc1_output))

        value = self.fc_value(fc2_output)
        advantage = self.fc_advantage(fc2_output).view(-1, self.action_size, self.atom_size)

        output = value + advantage - advantage.mean(1, keepdim=True)
        output = F.softmax(output, dim=-1)[0]

        return output
