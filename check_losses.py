import math
import torch
from losses import unsup_contrastive_loss, supcon_loss

B, D = 4, 16
torch.manual_seed(0)

random_z = torch.randn(2 * B, D)            # views unrelated: model has learned nothing
base = torch.eye(D)[:B]
perfect_z = torch.cat([base, base])         # views identical, other images orthogonal

for t in [1.0, 0.07]:
    print(f"tau={t}:  random {unsup_contrastive_loss(random_z, t):.3f}"
          f"   perfect {unsup_contrastive_loss(perfect_z, t):.4f}")
print("log(2B-1) =", round(math.log(2 * B - 1), 3))

labels = torch.tensor([0, 0, 1, 1] * 2)
print("supcon random:", round(supcon_loss(random_z, labels).item(), 3))