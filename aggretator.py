import numpy as np

# dy: advanced
def aggregate_grads(grads, backend):
    all_grads = {}
    n_users = len(grads)
    # n_smaples
    tot_sample = sum(gradinfo[0]['n_samples'] for gradinfo in grads)
    # avg_loss
    losses  = sorted([gradinfo[1].item() for gradinfo in grads])
    m = np.mean(losses)
    v = np.std(losses)
    losses  = [abs((i-m)/v) for i in losses]
    tot_loss = sum(losses) * v

    # AGG
    for _, gradinfo in enumerate(grads):
        feed = gradinfo[0]['n_samples'] / tot_sample
        feed += abs(gradinfo[1].item()-m) / tot_loss
        feed *= 0.5
        for k, v in gradinfo[0]['named_grads'].items():
            if k not in all_grads: all_grads[k] = []
            all_grads[k].append(v * feed)

    ret = {}
    for k, v in all_grads.items():
        ret[k] = backend.sum(v, dim=0)

    return [ret, grads[0][2]]

#***************************************************************************
def cal_by_mid(x, s, mid):
    if x>mid: return x+s
    return x-s

# dy: high loss contribute more
def aggregate_grads_dy(grads, backend):
    all_grads = {}
    # n_total_samples = 0
    n_users = len(grads)
    losses = sorted([gradinfo[1].item() for gradinfo in grads])
    s_loss = losses[0] * 0.95
    # b_loss = max(losses) - s_loss
    # tot_loss = sum((loss - s_loss) / b_loss for loss in losses)
    tot_loss = sum(losses)
    mid = losses[n_users>>1]

    # # print all param
    # print(len(grads[0][0]['named_grads'].keys()))
    # for nm in grads[0][0]['named_grads'].keys(): print(nm, end=', ')
    # print(' ')

    for _, gradinfo in enumerate(grads):
        # n_samples = gradinfo[0]['n_samples']
        # n_total_samples += n_samples
        feed = cal_by_mid(gradinfo[1].item(), s_loss, mid)
        for k, v in gradinfo[0]['named_grads'].items():
            if k not in all_grads: all_grads[k] = []
            all_grads[k].append(v * feed)

    gradients = {}
    # N = n_total_samples * tot_loss
    N = tot_loss
    for k, v in all_grads.items():
        gradients[k] = backend.sum(v, dim=0) / N

    return [gradients, grads[0][2]]

#***************************************************************************
def aggregate_grads_ori(grads, backend):
    total_grads = {}
    n_total_samples = 0
    for gradinfo in grads:
        n_samples = gradinfo[0]['n_samples']
        for k, v in gradinfo[0]['named_grads'].items():
            if k not in total_grads:
                total_grads[k] = []

            total_grads[k].append(v * n_samples)
        n_total_samples += n_samples

    gradients = {}
    for k, v in total_grads.items():
        gradients[k] = backend.sum(v, dim=0) / n_total_samples

    return [gradients, 0]