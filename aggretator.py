# dy: high loss contribute more
def aggregate_grads(grads, backend):
    all_grads = {}
    n_total_samples = 0
    n_users = len(grads)
    # sort by loss
    grads.sort(key=lambda x:x[1])
    N = (1 + n_users) * n_users * 0.5
    for i, gradinfo in enumerate(grads):
        n_samples = gradinfo[0]['n_samples']
        n_total_samples += n_samples
        for k, v in gradinfo[0]['named_grads'].items():
            if k not in all_grads: all_grads[k] = []
            all_grads[k].append(v * n_samples * (i+1.))

    gradients = {}
    n_total_samples *= N
    for k, v in all_grads.items():
        # v = sorted(v, key=lambda x:backend.torch.Tensor(x).norm())[-int(n_users * 0.5):]
        gradients[k] = backend.sum(v, dim=0) / n_total_samples

    return gradients

def aggregate_grads_ori(grads, backend):
    total_grads = {}
    n_total_samples = 0
    for gradinfo in grads:
        n_samples = gradinfo['n_samples']
        for k, v in gradinfo['named_grads'].items():
            if k not in total_grads:
                total_grads[k] = []

            total_grads[k].append(v * n_samples)
        n_total_samples += n_samples

    gradients = {}
    for k, v in total_grads.items():
        gradients[k] = backend.sum(v, dim=0) / n_total_samples

    return gradients