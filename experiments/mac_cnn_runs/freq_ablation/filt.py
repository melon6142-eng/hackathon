import numpy as np, torch
N=256
fy=np.fft.fftfreq(N)[:,None]; fx=np.fft.fftfreq(N)[None,:]
FR=np.sqrt(fx**2+fy**2)  # cycles/px
# bands (cycles/px): period >64px, 16-64, 4-16, <4
BANDS={'B1_low':(0,1/64),'B2_midlow':(1/64,1/16),'B3_midhigh':(1/16,1/4),'B4_high':(1/4,1.0)}
def soft(lo,hi,w=0.15):
    # smooth (log-freq cosine taper) band indicator
    f=np.maximum(FR,1e-6); lf=np.log2(f)
    def step(c):  # 0 below c, 1 above
        if c<=0: return np.ones_like(f)
        t=np.clip((lf-np.log2(c))/w*0.5+0.5,0,1); return 0.5-0.5*np.cos(np.pi*t)
    m=step(lo)*(1-step(hi) if hi<1 else 1)
    return m
MASK={k:soft(*v) for k,v in BANDS.items()}
def _apply(X,fn):
    x=X[:,0].double().numpy(); F=np.fft.fft2(x); dc=F[:,0,0].copy()
    F=fn(F); F[:,0,0]=dc
    o=np.real(np.fft.ifft2(F))
    return torch.tensor(np.clip(o,0,1),dtype=torch.float32)[:,None]
def remove(X,b): return _apply(X,lambda F:F*(1-MASK[b]))
def keep(X,b): return _apply(X,lambda F:F*MASK[b])
def phase_scramble(X,b,seed=0):
    rng=np.random.default_rng(seed)
    def fn(F):
        # random phase from FFT of white noise (Hermitian-safe)
        ph=np.angle(np.fft.fft2(rng.standard_normal(F.shape)))
        m=MASK[b]; return F*(1-m)+np.abs(F)*np.exp(1j*ph)*m
    return _apply(X,fn)
def noise(X,s,seed=0):
    g=torch.Generator().manual_seed(seed); return torch.clamp(X+torch.randn(X.shape,generator=g)*s,0,1)
VARIANTS=[('clean',lambda X:X)]+[(f'rm_{b}',(lambda b:lambda X:remove(X,b))(b)) for b in BANDS]\
 +[(f'keep_{b}',(lambda b:lambda X:keep(X,b))(b)) for b in BANDS]\
 +[(f'ps_{b}',(lambda b:lambda X:phase_scramble(X,b))(b)) for b in BANDS]\
 +[(f'noise_{s}',(lambda s:lambda X:noise(X,s))(s)) for s in (0.01,0.02,0.04,0.08,0.15)]
