import torch.nn as nn
import torch.nn.functional as F


class FLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(79, 256)
        self.fc5 = nn.Linear(256, 14)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc5(x)
        output = F.log_softmax(x, dim=1)
        return output

class ZTModel_1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size = 10),
            nn.ReLU(True)
        )
        
        self.fc1 = nn.Linear(4*70, 128)
        self.fc2 = nn.Linear(128,14)
        # i don't know whether multiple layers make sense
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(1, 8, kernel_size = 5),
#             nn.ReLU(True)
#         )
    
    def forward(self, x):
        in_size = x.size(0)
        x = self.layer1(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.layer2(x)
        output = F.log_softmax(x, dim = 1)
        return output