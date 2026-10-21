import numpy as np,pandas as pd
from sklearn.preprocessing import PolynomialFeatures
R='./open'
def base(sp):
    h=pd.read_csv(f'{R}/combo_v1/feat/hand_{sp}.csv').drop(columns='ID');t=pd.read_csv(f'{R}/combo_v1/feat/tex_{sp}.csv').drop(columns='ID')
    g=pd.read_csv(f'grain/grain_{sp}.csv').drop(columns=['ID','bnd_density']).fillna(0);r=pd.read_csv(f'raw/raw_{sp}.csv').drop(columns='ID')
    nb=h.shape[1]+t.shape[1]+g.shape[1]
    return np.hstack([h.values,t.values,g.values,r.values]).astype(float), nb, list(range(h.shape[1]))+list(range(h.shape[1]+t.shape[1],nb+r.shape[1]))
class PolyTop:
    """fold-internal top-k |corr| selection + poly2; uses only the fit() fold's labels"""
    def __init__(s,nb,cand,k=14): s.nb,s.cand,s.k=nb,cand,k
    def fit(s,X,y):
        C=X[:,s.cand]; c=np.abs([np.corrcoef(C[:,j],y)[0,1] if C[:,j].std()>0 else 0 for j in range(C.shape[1])])
        s.sel=[s.cand[j] for j in np.argsort(np.nan_to_num(c))[-s.k:]]; s.mu=X[:,s.sel].mean(0); s.sd=X[:,s.sel].std(0)+1e-9; return s
    def transform(s,X):
        Z=(X[:,s.sel]-s.mu)/s.sd; return np.hstack([X[:,:s.nb],PolynomialFeatures(2,include_bias=False).fit_transform(Z)])
