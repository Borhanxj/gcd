import torch

def ss_kmeans_single(x_l, y_l, x_u, k, num_old, g, n_iter=100, tol=1e-4):

    # Initialize centers with old labeled data
    old_centers = torch.stack([x_l[y_l == c].mean(0) for c in range(num_old)])

    # New centroids which is achieved by kmeans++ initialization on the unlabeled data
    centers = [old_centers]
    d2 = torch.cdist(x_u, old_centers).min(dim=1).values ** 2
    for _ in range(k-num_old):
        idx = torch.multinomial(d2 / d2.sum(), 1, generator=g)
        centers.append(x_u[idx])
        d2 = torch.minimum(d2, torch.cdist(x_u, x_u[idx]).squeeze(1) ** 2)
    centers = torch.cat(centers, dim=0)

    # Iterate such that labelled points are always forced to their original cluster
    x = torch.cat([x_l, x_u])
    for _ in range(n_iter):
        assign_u = torch.cdist(x_u, centers).argmin(dim=1)
        assign = torch.cat([y_l, assign_u])
        new_centers = torch.stack([
            x[assign == j].mean(0) if (assign == j).any() else centers[j]
            for j in range(k)])
        shift = (new_centers - centers).norm(dim=1).max()
        centers = new_centers
        if shift < tol:
            break

    assign_u = torch.cdist(x_u, centers).argmin(dim=1)
    assign = torch.cat([y_l, assign_u])
    inertia = ((x - centers[assign]) ** 2).sum()
    return assign_u, centers, inertia


def ss_kmeans(x_l, y_l, x_u, k, num_old, n_init=10, seed=0):
    g = torch.Generator(device=x_u.device).manual_seed(seed)
    best = None
    for _ in range(n_init):
        result = ss_kmeans_single(x_l, y_l, x_u, k, num_old, g)
        if best is None or result[2] < best[2]:
            best = result
    return best[0], best[1]