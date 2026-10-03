import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1)
        w = F.normalize(self.last_layer.weight, dim=1)   # unit-norm rows
        return F.linear(x, w)

class GCDModel(nn.Module):
    def __init__(self, grad_from_block=11, out_dim=65536):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        self.grad_from_block = grad_from_block

        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.blocks[grad_from_block:]:
            for p in blk.parameters():
                p.requires_grad = True

        self.head = ProjectionHead(768, out_dim)

    def features(self, x):
        b = self.backbone
        with torch.no_grad():                       # frozen part: no activations stored
            x = b.prepare_tokens(x)
            for blk in b.blocks[:self.grad_from_block]:
                x = blk(x)
        for blk in b.blocks[self.grad_from_block:]:  # trainable part
            x = blk(x)
        x = b.norm(x)
        return x[:, 0]                              # [CLS] token, shape [B, 768]

    def forward(self, x):
        feats = self.features(x)
        return feats, self.head(feats)