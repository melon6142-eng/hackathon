#!/usr/bin/env python3
"""Exp C: mild blur/noise sensitivity of frozen DINOv2 Ridge predictions (diagnostic only)."""
import os, time, numpy as np, torch, timm, torch.nn.functional as F, csv
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
R='~/AI/daker'; O=f'{R}/cnn/runs/exp_abc'
os.makedirs(O,exist_ok=True)
dev=os.environ.get('DEV','mps'); BS=int(os.environ.get('BS','8'))
print(f'== diag_c start {time.ctime()} dev={dev}', flush=True)
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
ARCH='vit_base_patch14_dinov2.lvd142m'; S=252
m=timm.create_model(ARCH,pretrained=True,num_classes=0,img_size=S).eval().to(dev)
for p in m.parameters(): p.requires_grad=False
dc=timm.data.resolve_data_config({},model=m)
mu=torch.tensor(dc['mean'],device=dev).view(1,3,1,1)
sd=torch.tensor(dc['std'],device=dev).view(1,3,1,1)
Xtr,_=torch.load(f'{R}/cnn/imgs.pt', map_location='cpu')
X=Xtr.float()
if X.max()>1.5: X=X/255.
if X.shape[1]==1: X=X.repeat(1,3,1,1)
N=len(X); print('X',tuple(X.shape), flush=True)

def gauss_blur(x, sigma):
    # separable approx via torchvision if available else stack blur via avg-ish F.conv2d
    if sigma<=0: return x
    k=int(max(3, round(sigma*6)//2*2+1))
    t=torch.arange(k,device=x.device,dtype=x.dtype)-k//2
    g=torch.exp(-0.5*(t/sigma)**2); g=g/g.sum()
    g=g.view(1,1,1,k).repeat(x.shape[1],1,1,1)
    pad=k//2
    y=F.pad(x,(pad,pad,0,0),mode='reflect')
    y=F.conv2d(y,g,groups=x.shape[1])
    g2=g.transpose(2,3)
    y=F.pad(y,(0,0,pad,pad),mode='reflect')
    y=F.conv2d(y,g2,groups=x.shape[1])
    return y

@torch.no_grad()
def feats(x):
    x=F.interpolate(x,size=S,mode='bilinear',align_corners=False)
    x=((x.to(dev)-mu)/sd)
    # last-layer CLS + patch mean (lightweight)
    out=m.forward_intermediates(x, indices=[len(m.blocks)-1], intermediates_only=True, output_fmt='NLC', norm=True, return_prefix_tokens=True)
    patch, prefix = out[0]
    return torch.cat([prefix[:,0], patch.mean(1)],1).float().cpu().numpy()

def extract_all(Ximg):
    outs=[]
    for i in range(0,len(Ximg),BS):
        outs.append(feats(Ximg[i:i+BS]))
    return np.concatenate(outs).astype(np.float32)

def mk():
    return make_pipeline(StandardScaler(), RidgeCV(np.logspace(0,5,21)))

def oof_pred(Fmat):
    o=np.zeros(len(y))
    for tr,va in KFold(5,shuffle=True,random_state=42).split(Fmat):
        o[va]=mk().fit(Fmat[tr],y[tr]).predict(Fmat[va])
    return o

t0=time.time()
print('extract clean...', flush=True)
Fc=extract_all(X)
print(f'clean feats {Fc.shape} {time.time()-t0:.0f}s', flush=True)
oc=oof_pred(Fc)
rmse=lambda a,b: float(np.sqrt(((a-b)**2).mean()))
hi=y>230
print(f'clean OOF RMSE {rmse(oc,y):.3f} hi_bias {(oc-y)[hi].mean():.1f}', flush=True)

# perturbations
cfgs=[
    ('blur_0.5', dict(blur=0.5, noise=0.0)),
    ('blur_1.0', dict(blur=1.0, noise=0.0)),
    ('blur_1.5', dict(blur=1.5, noise=0.0)),
    ('noise_0.01', dict(blur=0.0, noise=0.01)),
    ('noise_0.02', dict(blur=0.0, noise=0.02)),
    ('noise_0.05', dict(blur=0.0, noise=0.05)),
    ('blur1_noise0.02', dict(blur=1.0, noise=0.02)),
]
rows=[]
# Fit one full model on clean for transfer to perturbed (fairer sensitivity of fixed head)
full=mk().fit(Fc,y)
# Also OOF-style: use clean fold models applied to perturbed feats
fold_models=[]
for tr,va in KFold(5,shuffle=True,random_state=42).split(Fc):
    fold_models.append((va, mk().fit(Fc[tr],y[tr])))

for name,cfg in cfgs:
    t1=time.time()
    Xp=X.clone()
    if cfg['blur']>0:
        # blur on CPU then to avoid MPS issues with large
        chunks=[]
        for i in range(0,N,BS):
            xb=Xp[i:i+BS].to(dev)
            chunks.append(gauss_blur(xb, cfg['blur']).cpu())
        Xp=torch.cat(chunks)
    if cfg['noise']>0:
        g=torch.Generator().manual_seed(0)
        Xp=torch.clamp(Xp+torch.randn(Xp.shape,generator=g)*cfg['noise'],0,1)
    Fp=extract_all(Xp)
    # fixed-head (trained on clean) predictions
    p_full=full.predict(Fp)
    # fold-wise transfer
    p_oof=np.zeros(N)
    for va,mdl in fold_models:
        p_oof[va]=mdl.predict(Fp[va])
    d=p_oof-oc
    row=dict(
        name=name,
        mean_abs_d=float(np.abs(d).mean()),
        rmse_d=float(np.sqrt((d**2).mean())),
        corr_clean=float(np.corrcoef(p_oof,oc)[0,1]),
        oof_rmse=rmse(p_oof,y),
        delta_rmse=rmse(p_oof,y)-rmse(oc,y),
        hi_bias=float((p_oof-y)[hi].mean()),
        hi_mean_abs_d=float(np.abs(d[hi]).mean()),
        full_rmse_vs_clean=rmse(p_full,oc),
        t=time.time()-t1,
    )
    rows.append(row)
    print(f"{name}: |d|={row['mean_abs_d']:.3f} rmse_d={row['rmse_d']:.3f} corr={row['corr_clean']:.4f} "
          f"OOF={row['oof_rmse']:.3f} Δ={row['delta_rmse']:+.3f} hi|d|={row['hi_mean_abs_d']:.3f} ({row['t']:.0f}s)", flush=True)

import json
json.dump({'clean_oof_rmse':rmse(oc,y),'clean_hi_bias':float((oc-y)[hi].mean()),'rows':rows},
          open(f'{O}/diag_c.json','w'), indent=2)
np.save(f'{O}/oof/c_clean_oof.npy', oc)
print(f'DONE diag_c {time.time()-t0:.0f}s', flush=True)
