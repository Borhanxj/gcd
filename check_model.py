import torch
from model import GCDModel

def main():
    model = GCDModel().cuda()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    print(f"backbone params: {n_backbone/1e6:.1f}M  |  trainable total: {n_train/1e6:.1f}M")

    x = torch.randn(256, 3, 224, 224, device="cuda") # This is not the real data
    feats, proj = model(x)
    print("feats:", feats.shape, " proj:", proj.shape)

    proj.sum().backward()
    print("block 0 has grad:", model.backbone.blocks[0].attn.qkv.weight.grad is not None)
    print("block 11 has grad:", model.backbone.blocks[11].attn.qkv.weight.grad is not None)
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    torch.cuda.synchronize()
    print("1. GPU work finished")
    del model, x, feats, proj
    torch.cuda.empty_cache()
    print("2. memory released, exiting")

if __name__ == "__main__":
    main()
