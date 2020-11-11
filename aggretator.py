# dy: top-k norm only contribute
def aggregate_grads(grads, backend):
    all_grads = {}
    n_total_samples = 0
    n_users = len(grads)
    for gradinfo in grads:
        n_samples = gradinfo['n_samples']
        n_total_samples += n_samples
        for k, v in gradinfo['named_grads'].items():
            if k not in all_grads: all_grads[k] = []
            all_grads[k].append(v * n_samples)

    gradients = {}
    for k, v in all_grads.items():
        v = sorted(v, key=lambda x:backend.torch.Tensor(x).norm())[-(n_users>>1):]
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