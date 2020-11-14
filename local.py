import os

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F

from context import FederatedAveragingGrads
from context import PytorchModel
from learning_model import FLModel
from preprocess import CompDataset

_EPOCH      =   200
_BATCH_SIZE =   64
_BATCH_TEST =   10000
_LR         =   0.001
NUM_DATA    =   150000

def predict(model, data):
    model.eval()
    correct = 0
    loss = 0
    data_loader = torch.utils.data.DataLoader(data, batch_size=_BATCH_TEST)
    for _, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output  =  model(data)
        loss    += F.nll_loss(output, target.long(), reduction='sum').item()
        y_pred  =  output.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    loss /= len(data_loader.dataset)
    acc  = 100.00 * correct / len(data_loader.dataset)
    model.train()
    return acc, loss

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = FLModel().to(device)
    data = torch.load('lazy/URD_IID')
    choices = np.random.choice(len(data[0]), NUM_DATA)
    # print(len(choices), choices[:10])
    dataset = CompDataset(X=data[0][choices], Y=data[1][choices])
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=_BATCH_SIZE,
        shuffle=True,
    )
    print('Data read.')
    n_batch = len(train_loader)
    optim   =  torch.optim.Adam(model.parameters(), lr = _LR)

    print('Start Learning: batch_num-{}'.format(n_batch))
    model.train()
    for r in range(1, _EPOCH + 1):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target.long())
            total_loss += loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            pred = output.argmax(dim=1, keepdim=True)

        print('Round : {};  Avg loss: {:.4f}'.format(r, total_loss / n_batch))
        if r % 10 == 0:
            acc, loss = predict(model, dataset)
            print('\tACC: {:.3f};  Avg loss: {:.4f}'.format(acc, loss))