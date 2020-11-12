import torch
import torch.nn.functional as F

from preprocess import CompDataset


def user_round_train(X, Y, model, device, bs=320, debug=True):
    data = CompDataset(X=X, Y=Y)
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=bs,
        shuffle=True,
    )

    model.train()

    correct = 0
    prediction = []
    real = []
    total_loss = 0
    model = model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target.long(), reduction='sum')
        total_loss += loss
        loss.backward()
        pred = output.argmax(
            dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        prediction.extend(pred.reshape(-1).tolist())
        real.extend(target.reshape(-1).tolist())

    grads = {'n_samples': data.shape[0], 'named_grads': {}}
    for name, param in model.named_parameters():
        grads['named_grads'][name] = param.grad.detach().cpu().numpy()

    avg_loss = total_loss / len(train_loader.dataset)
    acc = 100. * correct / len(train_loader.dataset)
    if debug:
        print('Tot Loss: {:<8.2f}, Avg Loss: {:<10.4f}, Acc: {:<10.2f}'.format(total_loss, avg_loss, acc))

    return [grads, avg_loss]
