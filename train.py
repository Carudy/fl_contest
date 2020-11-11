import torch
import torch.nn.functional as F

from preprocess import CompDataset


def user_round_train(X, Y, model, device, debug=True):
    data = CompDataset(X=X, Y=Y)
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=320,
        shuffle=True,
    )

    model.train()

    correct = 0
    prediction = []
    real = []
    total_loss = 0
    model = model.to(device)
    n_batch = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # import ipdb
        # ipdb.set_trace()
        # print(data.shape, target.shape)

        # zt modified
        # data = data.reshape((320,1,79))
        # target = target.long()
        output = model(data)
        loss = F.nll_loss(output, target.long())
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

    avg_loss = total_loss / n_batch
    if debug:
        print('Total Loss: {:<8.2f}, Avg Loss: {:<8.2f}, Acc: {:<8.2f}'.format(
            total_loss, avg_loss, 100. * correct / len(train_loader.dataset)))

    return [grads, avg_loss]
