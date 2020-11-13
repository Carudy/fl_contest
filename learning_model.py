import torch.nn as nn
import torch.nn.functional as F

# zt: RNN mod
class FLModel_zt(nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        
        self.input = nn.Sequential(
            nn.BatchNorm1d(79),
            nn.Linear(79, 64),
            nn.ReLU(),
        )

        self.rnn64 = nn.RNN(input_size=64, hidden_size=32, nonlinearity='tanh', num_layers=2, dropout=0.02)
        self.rnn32 = nn.RNN(input_size=32, hidden_size=16, nonlinearity='tanh')
        self.bnn32 = nn.RNN(input_size=16, hidden_size=32, nonlinearity='tanh')
        self.bnn64 = nn.RNN(input_size=32, hidden_size=64, nonlinearity='tanh')
        
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

# dy: mod rnn
class FLModel_dy(nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        
        self.input = nn.Sequential(
            nn.BatchNorm1d(79),
            nn.Linear(79, 64),
            nn.ReLU(),
        )

        self.encoder = [
            nn.RNN(input_size=64, hidden_size=32, nonlinearity='relu', num_layers=2, dropout=0.01),
            nn.RNN(input_size=32, hidden_size=16, nonlinearity='relu'),
            nn.RNN(input_size=16, hidden_size=32, nonlinearity='relu'),
            nn.RNN(input_size=32, hidden_size=64, nonlinearity='relu'),
        ]
        
        self.output = nn.Sequential(
            nn.Linear(64, 14)
        )
        
    def forward(self, x): 
        x = self.input(x)
        x = x.reshape((1, x.shape[0], x.shape[1]))

        for net in self.encoder: x, _ = net(x)

        x = x.reshape(x.shape[1], x.shape[2])
        x = F.log_softmax(self.output(x), dim=1)
        return x


# dy: linear encoder
class FLModel(nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(79),
            nn.Linear(79, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            # nn.BatchNorm1d(16),
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
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x