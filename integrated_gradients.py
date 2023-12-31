import torch
import matplotlib.pyplot as plt
import numpy as np


def integrated_gradients(model, data, target, n_iter, device='cuda', visualize=True):
    assert isinstance(data, np.ndarray), "data should be np.ndarray type"
    assert data.ndim == 2, "(n_batch, n_feat)"

    input_x = torch.from_numpy(data).float().to(device)
    baseline = torch.from_numpy(np.random.random(size=(input_x.shape))).float().to(device)

    alpha = np.linspace(0, 1, n_iter)
    alpha = torch.from_numpy(alpha).float().to(device)
    alpha = alpha.view(n_iter, *tuple(np.ones(baseline[0].ndim, dtype='int')))

    attributions = []
    for i in range(input_x.shape[0]):
        x = input_x[i].unsqueeze(0)
        base = baseline[i].unsqueeze(0)
        scaled_x = base + alpha * (x - base)
        attribution = torch.zeros(*scaled_x.shape)
        for i in range(n_iter):
            part_scaled_x = scaled_x[i:i+1]
            part_scaled_x.requires_grad = True
            part_y_hat = model(part_scaled_x)
            # part_y_hat = part_y[:, target]
            attribution[i] = torch.autograd.grad(part_y_hat, part_scaled_x)[0]
        integrated = (attribution.sum(axis=0) / n_iter).to(device)
        ig = (x - base) * integrated
        ig = ig.detach().cpu().numpy().squeeze()
        attributions.append(ig)
    attributions = np.array(attributions)

    if visualize:
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(data.reshape(28, 28, 1), cmap='bone_r')
        plt.grid()
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(attributions.reshape(28, 28, 1), cmap='bone_r')
        plt.grid()
        plt.axis('off')
        plt.show()

    return attributions


def expected_gradients(model, data, baseline, target, n_iter, device='cuda', visualize=True):
    assert isinstance(data, np.ndarray),    "data should be np.ndarray type"
    assert data.ndim == 4, "(n_batch, n_feat, h, w)"
    assert isinstance(baseline, np.ndarray), "baseline should be np.ndarray type"
    assert baseline.ndim == 4, "(n_batch, n_feat, h, w)"

    input_x = torch.from_numpy(data).float().to(device)
    replace = baseline.shape[0] < n_iter
    sample_idx = np.random.choice(baseline.shape[0], size=(input_x.shape[0], n_iter), replace=replace)
    sampled_baseline = torch.from_numpy(baseline[sample_idx]).float().cuda()

    alpha = np.linspace(0, 1, n_iter)
    alpha = torch.from_numpy(alpha).float().to(device)
    alpha = alpha.view(n_iter, *tuple(np.ones(baseline[0].ndim, dtype='int')))

    attributions = []
    for i in range(input_x.shape[0]):
        x = input_x[i].unsqueeze(0)
        ref = sampled_baseline[i]
        scaled_x = ref + alpha * (x - ref)
        attribution = torch.zeros(*scaled_x.shape).to(device)
        for i in range(n_iter):
            part_scaled_x = scaled_x[i:i+1]
            part_scaled_x.requires_grad = True
            part_y_hat = model(part_scaled_x)[:, target]
            attribution[i] = torch.autograd.grad(part_y_hat, part_scaled_x)[0]
        integrated = attribution.sum(axis=0) / n_iter
        ig = (x - ref).mean(axis=0) * integrated
        ig = ig.detach().cpu().numpy().squeeze()
        attributions.append(ig)
    attributions = np.array(attributions)

    if visualize:
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(data.reshape(28, 28, 1), cmap='bone_r')
        plt.grid()
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(attributions.reshape(28, 28, 1), cmap='bone_r')
        plt.grid()
        plt.axis('off')
        plt.show()

    return attributions