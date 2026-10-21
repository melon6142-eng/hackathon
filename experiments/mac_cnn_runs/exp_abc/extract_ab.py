#!/usr/bin/env python3
"""Exp A: DINOv3 ViT-B/16 frozen CLS + mid/last patch mean/std
   Exp B: same forward — patch neighbor cosine/L2 distribution features
   Saves feats_a.npz, feats_b.npz under runs/exp_abc/
"""
import os, time, numpy as np, torch, timm, torch.nn.functional as F
R = '~/AI/daker'
O = f'{R}/cnn/runs/exp_abc'
os.makedirs(O, exist_ok=True)
dev = os.environ.get('DEV', 'mps')
ARCH = 'vit_base_patch16_dinov3.lvd1689m'
S = 256
BS = int(os.environ.get('BS', '8'))
NT = int(os.environ.get('NT', '4'))  # TTA: 0,1,2,3 rotations (4-way); set 8 for full
print(f'== extract_ab start {time.ctime()} arch={ARCH} S={S} BS={BS} NT={NT} dev={dev}', flush=True)

m = timm.create_model(ARCH, pretrained=True, num_classes=0).eval().to(dev)
for p in m.parameters():
    p.requires_grad = False
dc = timm.data.resolve_data_config({}, model=m)
mu = torch.tensor(dc['mean'], device=dev).view(1, 3, 1, 1)
sd = torch.tensor(dc['std'], device=dev).view(1, 3, 1, 1)
nb = len(m.blocks)
IDX = [nb // 2 - 1, nb - 1]  # mid, last  (5, 11)
npf = m.num_prefix_tokens
print(f'blocks={nb} IDX={IDX} n_prefix={npf} feat={m.num_features}', flush=True)

Xtr, Xte = torch.load(f'{R}/cnn/imgs.pt', map_location='cpu')
X = torch.cat([Xtr, Xte]).float()
if X.max() > 1.5:
    X = X / 255.
if X.shape[1] == 1:
    X = X.repeat(1, 3, 1, 1)
N = len(X)
print(f'X {tuple(X.shape)}', flush=True)

def d4(x, k):
    x = torch.rot90(x, k % 4, (2, 3))
    return torch.flip(x, (3,)) if k >= 4 else x

@torch.no_grad()
def forward_layers(x):
    # x: NCHW [0,1]
    x = F.interpolate(x, size=S, mode='bilinear', align_corners=False)
    x = ((x.to(dev) - mu) / sd)
    out = m.forward_intermediates(
        x, indices=IDX, intermediates_only=True, output_fmt='NLC',
        norm=True, return_prefix_tokens=True,
    )
    # list of (patch NLC, prefix NLP) per layer
    return out

def neighbor_stats(patch):
    """patch: (B, T, D) -> (B, F) neighbor cosine/L2 distribution features."""
    B, T, D = patch.shape
    g = int(T ** 0.5)
    assert g * g == T, (T, g)
    p = patch.reshape(B, g, g, D)
    qs = torch.tensor([0.1, 0.5, 0.9], device=patch.device)
    dirs = []
    dirs.append((p[:, :, :-1, :], p[:, :, 1:, :]))
    dirs.append((p[:, :-1, :, :], p[:, 1:, :, :]))
    dirs.append((p[:, :-1, :-1, :], p[:, 1:, 1:, :]))
    dirs.append((p[:, :-1, 1:, :], p[:, 1:, :-1, :]))
    out = []
    for a, b in dirs:
        a2 = a.reshape(B, -1, D)
        b2 = b.reshape(B, -1, D)
        # cosine
        an = a2 / (a2.norm(dim=-1, keepdim=True) + 1e-8)
        bn = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (an * bn).sum(-1)  # B,E
        dist = (a2 - b2).norm(dim=-1)  # B,E
        for arr in (cos, dist):
            out.append(arr.mean(1))
            out.append(arr.std(1))
            qv = torch.quantile(arr, qs, dim=1)  # 3,B
            out.extend([qv[i] for i in range(3)])
    return torch.stack(out, 1)  # B, 4dirs*2metrics*5 = 40

TT = list(range(NT)) if NT <= 4 else list(range(8))
print(f'TTA keys {TT}', flush=True)

# accumulators
A_keys = []
for j, li in enumerate(IDX):
    A_keys += [f'cls_L{j}', f'mean_L{j}', f'std_L{j}']
B_keys = [f'nb_L{j}' for j in range(len(IDX))]

acc_a = {k: [] for k in A_keys}
acc_b = {k: [] for k in B_keys}

t0 = time.time()
for i in range(0, N, BS):
    xb = X[i:i + BS]
    sum_a = None
    sum_b = None
    for k in TT:
        layers = forward_layers(d4(xb, k))
        fa = []
        fb = []
        for j, (patch, prefix) in enumerate(layers):
            # patch N,T,D ; prefix N,P,D — CLS = prefix[:,0]
            cls = prefix[:, 0]
            mean = patch.mean(1)
            std = patch.std(1)
            fa.extend([cls.float().cpu(), mean.float().cpu(), std.float().cpu()])
            fb.append(neighbor_stats(patch.float()).cpu())
        if sum_a is None:
            sum_a = [v / len(TT) for v in fa]
            sum_b = [v / len(TT) for v in fb]
        else:
            for ii in range(len(fa)):
                sum_a[ii] = sum_a[ii] + fa[ii] / len(TT)
            for ii in range(len(fb)):
                sum_b[ii] = sum_b[ii] + fb[ii] / len(TT)
    # store
    idx = 0
    for j in range(len(IDX)):
        for name in ('cls', 'mean', 'std'):
            acc_a[f'{name}_L{j}'].append(sum_a[idx].numpy())
            idx += 1
    for j in range(len(IDX)):
        acc_b[f'nb_L{j}'].append(sum_b[j].numpy())
    if i % (BS * 20) == 0 or i + BS >= N:
        print(f'  {i}/{N} {time.time()-t0:.0f}s', flush=True)

fa = {k: np.concatenate(v).astype(np.float32) for k, v in acc_a.items()}
fb = {k: np.concatenate(v).astype(np.float32) for k, v in acc_b.items()}
np.savez(f'{O}/feats_a.npz', **fa)
np.savez(f'{O}/feats_b.npz', **fb)
print('A shapes', {k: v.shape for k, v in fa.items()}, flush=True)
print('B shapes', {k: v.shape for k, v in fb.items()}, flush=True)
print(f'DONE extract_ab {time.time()-t0:.0f}s', flush=True)
