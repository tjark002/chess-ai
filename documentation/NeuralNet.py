import torch
import numpy as np

import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x

# Create an instance of the ChessNet
net = ChessNet()

# Define your input chess board
input_board = torch.randn(1, 64)  # Assuming a single input board

# Pass the input through the network
output = net(input_board)

# Print the output
print(output.item())  # Assuming a single output value
