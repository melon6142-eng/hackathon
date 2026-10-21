#!/usr/bin/env python3
"""BL1 DINO-branch proxy sensitivity (CPU).
BL1 = nested blend(HX, DINO L9-12 mean+std). No saved Ridge/NNLS weights.
ViT-B FT members: no fold weight checkpoints (fold.npz = va/pv/pt only).
This script ONLY proxies the frozen-DINO half of BL1 with a clean-fit Ridge head
applied to perturbed features — labeled as proxy, not true fold-model inference.
"""
import os, time, json, csv, numpy as np, torch, timm, torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.ndimage import gaussian_filter

R='~/AI/daker'; O=f'{R}/cnn/runs/robust_pairs'
os.makedirs(O,exist_ok=True)
dev='cpu'  # avoid contending with dino12 MPS
BS=8
print(f'== proxy_bl1_dino {time.ctime()} dev={dev}', flush=True)
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
hi=y>230; lo=y<170
rmse=lambda a,b: float(np.sqrt(((a-b)**2).mean()))

# Checkpoint inventory
inv=[]
members=['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2','dinoB_trans','dinoB_trans_s1','dinoB_trans_s2']
for t in members:
    p=f'{R}/cnn/runs/{t}'
    files=sorted(os.listdir(p)) if os.path.isdir(p) else []
    has_w=any(x.endswith(('.pt','.pth','.ckpt')) for x in files)
    fold=np.load(f'{p}/fold0.npz') if os.path.exists(f'{p}/fold0.npz') else None
    inv.append(dict(tag=t, files=files, has_weight_ckpt=has_w,
                    fold0_keys=list(fold.files) if fold is not None else None))
inv.append(dict(tag='BL1', has_weight_ckpt=False,
                note='oof_blend1.npy only; HX+DINO Ridge/NNLS weights not serialized'))
json.dump(inv, open(f'{O}/ckpt_inventory.json','w'), indent=2)
print('inventory', json.dumps(inv, indent=2)[:800], flush=True)

# Load images (train only for OOF proxy)
Xtr,_=torch.load(f'{R}/cnn/imgs.pt', map_location='cpu')
X=Xtr.float().numpy()
if X.max()>1.5: X=X/255.
if X.ndim==4 and X.shape[1]==1: X=X[:,0]
N=len(X)

# Frozen DINOv2 — match BL1 spirit: last-4 layers mean+std of patches (approx L9-12)
# BL1 used vitb14_252 multilayer mean+std; we use forward_intermediates indices for blocks 8,9,10,11
ARCH='vit_base_patch14_dinov2.lvd142m'; S=252
m=timm.create_model(ARCH, pretrained=True, num_classes=0, img_size=S).eval()
for p in m.parameters(): p.requires_grad=False
dc=timm.data.resolve_data_config({}, model=m)
mu=torch.tensor(dc['mean']).view(1,3,1,1)
sd=torch.tensor(dc['std']).view(1,3,1,1)
IDX=[8,9,10,11]
print('model ready', flush=True)

@torch.no_grad()
def feats_batch(imgs_np):
    # imgs_np: B,H,W in [0,1]
    x=torch.from_numpy(imgs_np).float()[:,None].repeat(1,3,1,1)
    x=F.interpolate(x, size=S, mode='bilinear', align_corners=False)
    x=(x-mu)/sd
    out=m.forward_intermediates(x, indices=IDX, intermediates_only=True, output_fmt='NLC',
                                norm=True, return_prefix_tokens=True)
    parts=[]
    for patch, prefix in out:
        parts += [patch.mean(1), patch.std(1)]
    return torch.cat(parts,1).numpy().astype(np.float32)

def extract(Ximg):
    outs=[]
    for i in range(0,len(Ximg),BS):
        outs.append(feats_batch(Ximg[i:i+BS]))
        if i%(BS*25)==0: print('  feat', i, flush=True)
    return np.concatenate(outs)

def perturb(Ximg, blur=0.0, noise=0.0, seed=0):
    out=Ximg.copy()
    if blur>0:
        for i in range(len(out)):
            out[i]=gaussian_filter(out[i], sigma=blur)
    if noise>0:
        rng=np.random.default_rng(seed)
        out=np.clip(out+rng.normal(0, noise, out.shape).astype(np.float32), 0, 1)
    return out.astype(np.float32)

mk=lambda: make_pipeline(StandardScaler(), RidgeCV(np.logspace(0,5,21)))

def oof_apply(Fc, Fp):
    """Fit Ridge per fold on clean Fc, predict on Fp[va]."""
    o_clean=np.zeros(N); o_pert=np.zeros(N)
    for tr,va in KFold(5, shuffle=True, random_state=42).split(Fc):
        mdl=mk().fit(Fc[tr], y[tr])
        o_clean[va]=mdl.predict(Fc[va])
        o_pert[va]=mdl.predict(Fp[va])
    return o_clean, o_pert

t0=time.time()
print('extract clean...', flush=True)
Fc=extract(X)
print(f'clean {Fc.shape} {time.time()-t0:.0f}s', flush=True)

cfgs=[
    ('clean', 0.0, 0.0),
    ('blur_0.5', 0.5, 0.0),
    ('blur_1.0', 1.0, 0.0),
    ('noise_0.005', 0.0, 0.005),
    ('noise_0.01', 0.0, 0.01),
    ('noise_0.02', 0.0, 0.02),
]
rows=[]
oc_ref=None
for name,b,n in cfgs:
    t1=time.time()
    if name=='clean':
        Fp=Fc
    else:
        Xp=perturb(X, blur=b, noise=n, seed=0)
        Fp=extract(Xp)
    oc, op = oof_apply(Fc, Fp)
    if oc_ref is None: oc_ref=oc
    d=op-oc_ref
    row=dict(name=name, blur=b, noise=n,
             mean_abs_d=float(np.abs(d).mean()),
             rmse_d=float(np.sqrt((d**2).mean())),
             corr_clean=float(np.corrcoef(op, oc_ref)[0,1]),
             oof_rmse=rmse(op,y),
             delta_rmse=rmse(op,y)-rmse(oc_ref,y),
             hi_bias=float((op-y)[hi].mean()),
             hi_mean_abs_d=float(np.abs(d[hi]).mean()),
             lo_mean_abs_d=float(np.abs(d[lo]).mean()),
             t=time.time()-t1)
    rows.append(row)
    print(f"{name}: |d|={row['mean_abs_d']:.3f} rmse_d={row['rmse_d']:.3f} corr={row['corr_clean']:.4f} "
          f"OOF={row['oof_rmse']:.3f} Δ={row['delta_rmse']:+.3f} hi|d|={row['hi_mean_abs_d']:.3f}", flush=True)
    np.save(f'{O}/proxy_{name}_oof.npy', op)

# Multi-seed stability of sensitivity (fold seeds 42,100,101): only for blur_0.5 and noise_0.01
stab=[]
for name,b,n in [('blur_0.5',0.5,0.0),('noise_0.01',0.0,0.01)]:
    Xp=perturb(X, blur=b, noise=n, seed=0)
    Fp=extract(Xp) if name!='reuse' else None
    # Fp already extracted above — reload from re-extract for clarity
    Fp=extract(Xp)
    stats=[]
    for rs in [42,100,101,102,103]:
        o_c=np.zeros(N); o_p=np.zeros(N)
        for tr,va in KFold(5,shuffle=True,random_state=rs).split(Fc):
            mdl=mk().fit(Fc[tr],y[tr])
            o_c[va]=mdl.predict(Fc[va]); o_p[va]=mdl.predict(Fp[va])
        d=o_p-o_c
        stats.append(dict(rs=rs, mean_abs_d=float(np.abs(d).mean()),
                          corr=float(np.corrcoef(o_p,o_c)[0,1]),
                          delta_rmse=rmse(o_p,y)-rmse(o_c,y)))
    stab.append(dict(name=name,
                     mean_abs_d_mean=float(np.mean([s['mean_abs_d'] for s in stats])),
                     mean_abs_d_std=float(np.std([s['mean_abs_d'] for s in stats])),
                     corr_mean=float(np.mean([s['corr'] for s in stats])),
                     corr_std=float(np.std([s['corr'] for s in stats])),
                     delta_rmse_mean=float(np.mean([s['delta_rmse'] for s in stats])),
                     delta_rmse_std=float(np.std([s['delta_rmse'] for s in stats])),
                     per_seed=stats))
    print('stab', name, stab[-1], flush=True)

out=dict(component='BL1_DINO_branch_proxy',
         disclaimer='No fold weight ckpt. Clean-fit Ridge on frozen DINOv2 L9-12 mean+std → apply to perturbed feats. Not HX half, not true BL1 NNLS blend.',
         clean_oof_rmse=rmse(oc_ref,y), clean_hi_bias=float((oc_ref-y)[hi].mean()),
         rows=rows, stability=stab, inventory_file='ckpt_inventory.json')
json.dump(out, open(f'{O}/proxy_bl1_dino.json','w'), indent=2)
print(f'DONE proxy {time.time()-t0:.0f}s', flush=True)
