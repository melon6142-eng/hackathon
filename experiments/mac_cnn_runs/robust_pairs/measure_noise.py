#!/usr/bin/env python3
"""Measure high-frequency residual / Laplacian / sharpness on all 1500 images. CPU only."""
import os, time, json, numpy as np, torch, csv
from scipy.ndimage import gaussian_filter, laplace
R='~/AI/daker'; O=f'{R}/cnn/runs/robust_pairs'
os.makedirs(O,exist_ok=True)
print(f'== measure_noise {time.ctime()}', flush=True)
Xtr,Xte=torch.load(f'{R}/cnn/imgs.pt', map_location='cpu')
X=torch.cat([Xtr,Xte]).float().numpy()
if X.max()>1.5: X=X/255.
if X.ndim==4 and X.shape[1]==1: X=X[:,0]
elif X.ndim==4: X=X.mean(1)
N=len(X); assert X.shape==(1500,256,256), X.shape
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
# metrics per image
hf_std=np.zeros(N); lap_var=np.zeros(N); sharp=np.zeros(N); pix_std=np.zeros(N)
t0=time.time()
for i in range(N):
    img=X[i]
    blur=gaussian_filter(img, sigma=1.0)
    res=img-blur
    hf_std[i]=res.std()
    L=laplace(img)
    lap_var[i]=L.var()
    # Tenengrad-like sharpness: mean squared gradient magnitude
    gy,gx=np.gradient(img)
    sharp[i]=np.mean(gx*gx+gy*gy)
    pix_std[i]=img.std()
    if i%300==0: print(i, f'{time.time()-t0:.0f}s', flush=True)

def summ(a, name):
    q=np.quantile(a,[0,0.05,0.25,0.5,0.75,0.95,1.0])
    return dict(name=name, mean=float(a.mean()), std=float(a.std()),
                q=dict(zip(['min','p5','p25','p50','p75','p95','max'], map(float,q))))

# Injected noise scales in [0,1] image space: 0.005, 0.01, 0.02
# Compare to hf_std distribution
inj_noise=[0.005,0.01,0.02]
inj_blur=[0.5,1.0]
# For blur: measure HF change at those sigmas on a sample
sample=np.random.default_rng(0).choice(N, 100, replace=False)
blur_effect=[]
for sig in inj_blur:
    d=[]
    for i in sample:
        b=gaussian_filter(X[i], sigma=sig)
        d.append(dict(hf_std_after=float((X[i]-gaussian_filter(b,1.0)).std()),
                      hf_drop=float(hf_std[i]-(X[i]-gaussian_filter(b,1.0)).std()),
                      sharp_after=float(np.mean(np.gradient(b)[0]**2+np.gradient(b)[1]**2)),
                      sharp_ratio=float(np.mean(np.gradient(b)[0]**2+np.gradient(b)[1]**2)/(sharp[i]+1e-12))))
    blur_effect.append(dict(sigma=sig,
        mean_sharp_ratio=float(np.mean([z['sharp_ratio'] for z in d])),
        mean_hf_drop=float(np.mean([z['hf_drop'] for z in d])),
        p50_sharp_ratio=float(np.median([z['sharp_ratio'] for z in d]))))

# Where does injected noise sit vs natural hf_std?
# Additive Gaussian noise with σ=n increases residual roughly by ~n (if HF dominated by noise)
noise_vs_data=[]
for n in inj_noise:
    # percentile of natural hf_std that equals n
    pct=float((hf_std<=n).mean()*100)
    # ratio n / median(hf_std)
    noise_vs_data.append(dict(sigma=n, pct_images_hf_std_le=pct,
                              vs_median_hf=float(n/np.median(hf_std)),
                              vs_p5_hf=float(n/np.quantile(hf_std,0.05)),
                              vs_p95_hf=float(n/np.quantile(hf_std,0.95)),
                              in_typical_band=bool(np.quantile(hf_std,0.05)<=n<=np.quantile(hf_std,0.95))))

# train vs test
split=dict(train={k:summ(a[:500],k) for k,a in [('hf_std',hf_std),('lap_var',lap_var),('sharp',sharp),('pix_std',pix_std)]},
           test={k:summ(a[500:],k) for k,a in [('hf_std',hf_std),('lap_var',lap_var),('sharp',sharp),('pix_std',pix_std)]},
           all={k:summ(a,k) for k,a in [('hf_std',hf_std),('lap_var',lap_var),('sharp',sharp),('pix_std',pix_std)]})

# high-hardness subset
hi=np.zeros(N,dtype=bool); hi[:500]=y>230
split['hi230_train']={k:summ(a[hi],k) for k,a in [('hf_std',hf_std),('lap_var',lap_var),('sharp',sharp),('pix_std',pix_std)]}

out=dict(n=N, blur_effect_sample100=blur_effect, noise_vs_data=noise_vs_data, split=split,
         arrays_path='noise_arrays.npz')
np.savez(f'{O}/noise_arrays.npz', hf_std=hf_std.astype(np.float32), lap_var=lap_var.astype(np.float32),
         sharp=sharp.astype(np.float32), pix_std=pix_std.astype(np.float32))
json.dump(out, open(f'{O}/noise_stats.json','w'), indent=2)
print(json.dumps(dict(all_hf=split['all']['hf_std'], noise_vs=noise_vs_data, blur=blur_effect), indent=2), flush=True)
print(f'DONE measure_noise {time.time()-t0:.0f}s', flush=True)
