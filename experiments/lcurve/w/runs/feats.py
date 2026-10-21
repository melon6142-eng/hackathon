import numpy as np,pandas as pd
from sklearn.preprocessing import PolynomialFeatures
R='./open'
def hx(sp):
    h=pd.read_csv(f'{R}/combo_v1/feat/hand_{sp}.csv').drop(columns='ID');t=pd.read_csv(f'{R}/combo_v1/feat/tex_{sp}.csv').drop(columns='ID').values
    g=pd.read_csv(f'grain/grain_{sp}.csv').drop(columns=['ID','bnd_density']).fillna(0);r=pd.read_csv(f'raw/raw_{sp}.csv').drop(columns='ID')
    return h,t,g,r
TOP=None
def HX(sp):
    """P1 feature set: hand+tex+grain + poly2 of 14 top-|corr| cols chosen on TRAIN only (fixed list, standardized with train stats)"""
    h,t,g,r=hx(sp); htr,_,gtr,rtr=hx('train'); Y=pd.read_csv(f'{R}/train.csv').hardness
    top=pd.concat([htr,gtr,rtr],axis=1); c=top.corrwith(Y).abs().sort_values().tail(14).index
    mu,sd=top[c].mean(),top[c].std()
    Z=((pd.concat([h,g,r],axis=1)[c]-mu)/sd).values
    P=PolynomialFeatures(2,include_bias=False).fit_transform(Z)
    return np.hstack([h.values,t,g.values,P])
def DINO(sp,name='vitb14_252',sel=None):
    F=np.load(f'dino/{name}_{sp}.npy',mmap_mode='r'); D=F.shape[1]//16 if sel else None
    return np.hstack([np.asarray(F[:,(L*4+k)*D:(L*4+k+1)*D]) for L,k in sel]) if sel else np.asarray(F)
