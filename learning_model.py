import torch.nn as nn
import torch.nn.functional as F

class FLModel_zyc(nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(79, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
          
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),

            nn.Linear(16, 14)
            )

    def forward(self, x):
        x = self.layer(x)
        output = F.log_softmax(x, dim=1)
        return output


# dy: encoder
class FLModel_dy(nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(78),
            nn.Linear(78, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Linear(64, 14)
        )
        
    def forward(self, x):
        x = x[:, :78]
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x


# zt: RNN mod
class FLModel(nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        
        self.input = nn.Sequential(
            nn.BatchNorm1d(79),
            nn.Linear(79, 64),
            nn.ReLU(),
        )

        self.rnn64 = nn.RNN(input_size=64, hidden_size=32)
        self.rnn32 = nn.RNN(input_size=32, hidden_size=16)
        # self.rnn16 = nn.RNN(input_size=16, hidden_size=8)
        # self.bnn16 = nn.RNN(input_size=8,  hidden_size=16)
        self.bnn32 = nn.RNN(input_size=16, hidden_size=32)
        self.bnn64 = nn.RNN(input_size=32, hidden_size=64)
        
        self.output = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(64, 14)
        )
        
    def forward(self, x):
        x = self.input(x)
        x = x.reshape((1, x.shape[0], x.shape[1]))
        x, _ = self.rnn64(x)
        x, _ = self.rnn32(x)
        # x, _ = self.rnn16(x)
        # x, _ = self.bnn16(x)
        x, _ = self.bnn32(x)
        x, _ = self.bnn64(x)
        x = x.reshape(x.shape[1], x.shape[2])
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x