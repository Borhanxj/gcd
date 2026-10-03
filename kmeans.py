import torch

def kmeans_pp_init(x, k, g):
    n = x.shape[0]

    first = torch.randint(n, (1,), generator=g, device=x.device)
    centers = [x[first]]
    d2 = torch.cdist(x, centers[0]).squeeze(1) ** 2 

    for _ in range(1, k):
        idx = torch.multinomial(d2 / d2.sum(), 1, generator=g)
        centers.append(x[idx])
        d2 = torch.minimum(d2, torch.cdist(x, x[idx]).squeeze(1) ** 2)

    return torch.cat(centers, dim=0)

def kmeans_single(x, k, g, n_iter=100, tol=1e-4):
    centers = kmeans_pp_init(x, k, g)

    for _ in range(n_iter):
        # Assign each point to the nearest center
        assign = torch.cdist(x, centers).argmin(dim=1)

        # Update centers
        new_centers = torch.stack([
            x[assign == i].mean(0) if (assign == i).any() else centers[i]
            for i in range(k)
        ]) 

        # Check for convergence
        shift = (new_centers - centers).norm(dim=1).max()
        centers = new_centers

        if shift < tol:
            break

    assign = torch.cdist(x, centers).argmin(dim=1)
    inertia = ((x - centers[assign]) ** 2).sum() 
    return assign, centers, inertia


def kmeans(x, k, n_init=10, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    best = None

    for _ in range(n_init):
        result = kmeans_single(x, k, g)
        if best is None or result[2] < best[2]:
            best = result
    return best[0], best[1]