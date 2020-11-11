import os

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F

from context import FederatedAveragingGrads
from context import PytorchModel
from learning_model import FLModel
from preprocess import CompDataset

_EPOCH      =   100
_BATCH_SIZE =   320
_BATCH_TEST =   1200
_LR         =   0.01

def predict(model, data):
    model.eval()
    correct = 0
    loss = 0
    data_loader = torch.utils.data.DataLoader(data, batch_size=_BATCH_TEST)
    for _, (data, target) in enumerate(data_loader):
        output  =  model(data)
        loss    += F.nll_loss(output, target.long(), reduction='sum').item()
        y_pred  =  output.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    loss /= len(data_loader.dataset)
    acc  = 100.00 * correct / len(data_loader.dataset)
    model.train()
    return acc, loss

if __name__ == '__main__':
    model = FLModel()
    data = torch.load('lazy/URD_IID')
    print('Data read.')
    dataset = CompDataset(X=data[0], Y=data[1])
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=_BATCH_SIZE,
        shuffle=True,
    )
    n_batch = len(train_loader)
    optim   =  torch.optim.Adam(model.parameters(), lr = _LR)

    print('Start Learning:')
    model.train()
    for r in range(1, _EPOCH + 1):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = F.nll_loss(output, target.long())
            total_loss += loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            pred = output.argmax(dim=1, keepdim=True)

        print('Round : {};  Avg loss: {:.4f}'.format(r, total_loss / n_batch))
        if r<10:
            acc, loss = predict(model, dataset)
            print('\tACC: {:.3f};  Avg loss: {:.4f}'.format(acc, loss))