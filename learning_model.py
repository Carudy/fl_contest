import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing

class FLModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(79, 256)
        # self.fc5 = nn.Linear(256, 14)
        self.layer = nn.Sequential(
            nn.Linear(79, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Dropout(0.2),

            nn.Linear(16, 14)
            )

    def forward(self, x):
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc5(x)
        # print(x.shape)

        x = self.layer(x)
        output = F.log_softmax(x, dim=1)
        # print(output.shape)

        return output
