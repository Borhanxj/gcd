import torch
import torch.nn.functional as F

def unsup_contrastive_loss(z, temperature=1.0):
    # given 2 augmented views of B images, z: [2B, D], compute the unsupervised contrastive loss
    # parameter z is of shape [2B, D] where B is the batch size and D is the feature dimension. 
    # The first B rows correspond to the first view and the next B rows correspond to the second view of the same images.
    z = F.normalize(z, dim=-1)
    n, B = z.shape[0], z.shape[0] // 2

    # compute similarity matrix
    sim = z @ z.T / temperature 
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    targets = torch.cat([torch.arange(B, n), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)

def supcon_loss(z, labels, temperature=0.07):
    # given labels for 2 augmented views of B images, z: [2B, D], compute the supervised contrastive loss
    # Only for labelled images.
    z = F.normalize(z, dim=-1)
    n = z.shape[0]

    # Compute the similarity matrix.
    sim = z @ z.T / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask  # same class, not self
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)    # log-softmax per row
    mean_log_prob_pos = log_prob.masked_fill(~pos_mask, 0).sum(1) / pos_mask.sum(1)
    return -mean_log_prob_pos.mean()

def gcd_loss(proj, labels, is_lab, lam=0.35, t_u=1.0, t_s=0.07):
    # proj: [2B, D]; labels, is_lab: [B] (labels are -1 for unlabelled images)
    l_u = unsup_contrastive_loss(proj, t_u)                       # ALL images

    p1, p2 = proj.chunk(2)
    z_l = torch.cat([p1[is_lab], p2[is_lab]])                     # labelled only
    y_l = torch.cat([labels[is_lab], labels[is_lab]])
    l_s = supcon_loss(z_l, y_l, t_s)

    loss = (1 - lam) * l_u + lam * l_s
    return loss, l_u.detach(), l_s.detach()