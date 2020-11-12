# dy: high loss contribute more
def aggregate_grads(grads, backend):
    all_grads = {}
    n_total_samples = 0
    tot_loss = sum(gradinfo[1].item() for gradinfo in grads)
    n_users = len(grads)

    # sort by loss
    # print(tot_loss)
    # grads.sort(key=lambda x:-x[1])
    # for i in range(n_users):
    #     for k, v in grads[i][0]['named_grads'].items(): 
    #         v *= 2. if i < n_users*0.3 else (0.1 if i > n_users*0.8 else 1.)

    for _, gradinfo in enumerate(grads):
        n_samples = gradinfo[0]['n_samples']
        n_total_samples += n_samples
        n_samples *=gradinfo[1].item()
        for k, v in gradinfo[0]['named_grads'].items():
            if k not in all_grads: all_grads[k] = []
            all_grads[k].append(v * n_samples)

    gradients = {}
    N = n_total_samples * tot_loss
    for k, v in all_grads.items():
        gradients[k] = backend.sum(v, dim=0) / N

    return [gradients, grads[0][2]]

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