import torch
import torch.nn as nn


# Define the model in PyTorch
class ChessNet(nn.Module):
    def __init__(self, hidden_size):
        super(ChessNet, self).__init__()
        inchannels = 14

        # Convolutional Layers
        self.conv1 = nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1)

        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(inchannels)
        self.bn2 = nn.BatchNorm2d(inchannels)

        # Activation Layers
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

        # Fully Connected Layers
        self.fc = nn.Linear(14 * 8 * 8, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        #print(f"Input Shape: {x.shape}")
        x = x.to(torch.float32)
        x_input = x.clone()

        # Convolutional Block 1
        x = self.conv1(x)
        #print(f"After Conv1: {x.shape}")
        x = self.bn1(x)
        #print(f"After BN1: {x.shape}")
        x = self.activation1(x)
        #print(f"After Activation1: {x.shape}")

        # Convolutional Block 2
        x = self.conv2(x)
        #print(f"After Conv2: {x.shape}")
        x = self.bn2(x)
        #print(f"After BN2: {x.shape}")

        # Residual Connection
        x = x + x_input
        x = self.activation2(x)
        #print(f"After Residual Connection: {x.shape}")

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        #print(f"After Flatten: {x.shape}")
        x = torch.relu(self.fc(x))
        #print(f"After FC: {x.shape}")
        x = self.output_layer(x)
        #print(f"After Output Layer: {x.shape}")

        return x.squeeze()