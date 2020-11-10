def direction_grads(a, b):
    if a*b > 1e-8:
        return a
    else:
        return 0

def aggregate_grads(grads, backend):
    all_grads = {}
    n_total_samples = 0
    for gradinfo in grads:
        for k, v in gradinfo['named_grads'].items():
            if k not in all_grads: all_grads[k] = []
            all_grads[k].append(v)

    gradients = {}
    for k, v in all_grads.items():
        v = backend.torch.Tensor(v)
        frac = 1.0 / len(v)
        d = (((v>1e-8)*1.0) + ((v<-1e-8)*-1.0)).sum(dim=0)
        for son in v:
            son.map_(d, direction_grads)
        gradients[k] = v.sum(dim=0)

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