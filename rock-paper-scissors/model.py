import torch
import torch.nn as nn
import torch.optim as optim
class MyModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out