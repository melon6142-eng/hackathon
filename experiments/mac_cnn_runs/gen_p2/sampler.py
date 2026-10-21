import numpy as np,json,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'../gen_cf'))
import gen as G0
PR=dict(G0.PRIOR,tex_a=(0,.05,0),tex_s=(.6,1.6,0),sharp=(0,1.2,0),pore_d=(0.05,0.6,0))
TGT=['n_grains','aspect','warp','ph_frac','pore_frac','bnd_c','gs_het','ph_clust','tex_a']
def tgt(p): return np.array([np.log(p['n_grains']),np.log(p['aspect']),p['warp'],p['ph_frac'],p['pore_frac'],p['bnd_c'],p['gs_het'],p['ph_clust'],p['tex_a']],np.float32)
def sample(P5,idx,n,seed):
    r=np.random.default_rng(seed);out=[]
    for _ in range(n):
        p=dict(P5[int(r.choice(idx))])
        for k,(lo,hi,lg) in PR.items():
            if k not in p: continue
            if k=='theta': p[k]=float(r.uniform(0,np.pi));continue
            v=p[k]*np.exp(r.normal(0,.15)) if lg else p[k]+r.normal(0,.08*(hi-lo))
            p[k]=float(np.clip(v,lo,hi))
        p['gs_het']=float(r.uniform(0,.8));p['ph_clust']=float(r.uniform(0,.8));p['pore_clust']=float(r.uniform(0,.8))
        out.append(p)
    return out
