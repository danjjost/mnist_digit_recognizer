import torch

import torch.nn as nn
import torch.nn.functional as F

class MNISTDigitRecognizerNeuralNet(nn.Module):

    def __init__(self, debug: bool = False) -> None:
        super().__init__()  # type: ignore
        self.debug = debug
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3) # 3x3 kernels, moving across the 1 input channels (greyscale), outputting 7 channels. The new (7) channels will be 26x26
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 7 feature maps, 13x13

        # 8 
        self.fc1 = nn.Linear(8*13*13, 10)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolve and Pool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = torch.flatten(x, 1)

        # Fully Connected Layer
        x = self.fc1(x)
        #x = F.relu(x)
        x = F.log_softmax(x, dim=1)

        # Output Layer
        #x = self.fc2(x)
        #x = F.log_softmax(x, dim=1)

        return x