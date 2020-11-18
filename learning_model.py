import torch.nn as nn
import torch.nn.functional as F

# zt: RNN mod
class FLModel(nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        
        self.input = nn.Sequential(
            nn.BatchNorm1d(79),
            nn.Linear(79, 64),
            nn.ReLU(),
        )

        self.rnn64 = nn.LSTM(input_size=64, hidden_size=32)
        self.rnn32 = nn.LSTM(input_size=32, hidden_size=16)
        self.bnn32 = nn.LSTM(input_size=16, hidden_size=32)
        self.bnn64 = nn.LSTM(input_size=32, hidden_size=64)
        
        self.output = nn.Sequential(
            nn.Linear(64, 14)
        )
        
    def forward(self, x): 
        x = self.input(x)
        x = x.reshape((1, x.shape[0], x.shape[1]))
        x, _ = self.rnn64(x)
        x, _ = self.rnn32(x)
        x, _ = self.bnn32(x)
        x, _ = self.bnn64(x)
        x = x.reshape(x.shape[1], x.shape[2])
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x

# dy: simple
class FLModel_sim(nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        
        self.net = nn.Sequential(
            nn.BatchNorm1d(79),
            nn.Linear(79, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 14),
        )

    def forward(self, x): 
        x = F.log_softmax(self.net(x), dim=1)
        return x


# dy: linear encoder
class FLModel_line(nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        self.autoencoder = nn.Sequential(
            nn.BatchNorm1d(79),
            nn.Linear(79, 64),
            nn.ReLU(),
            # encode
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            # decode
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Tanh(),
        )

        self.output = nn.Sequential(
            nn.Linear(64, 14)
        )
        
    def forward(self, x):
        x = self.autoencoder(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x