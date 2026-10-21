import os;os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,sys
from skimage import io,exposure,filters
from joblib import Parallel,delayed
R='./open'
def f(p):
    im=io.imread(p,as_gray=True).astype(float); im=im/255 if im.max()>1 else im
    out={}
    for nm,t in [('cl',exposure.equalize_adapthist(im,clip_limit=0.02)),('g2',im**2.0),('st',exposure.rescale_intensity(im,in_range=tuple(np.percentile(im,(1,99))),out_range=(0,1)))]:
        s=filters.gaussian(t,1); g=filters.sobel(s); th=filters.threshold_otsu(s); dark=s<th
        q=np.percentile(s,[1,5,25,50,75,95,99]); 
        for i,v in zip([1,5,25,50,75,95,99],q): out[f'{nm}_q{i}']=v
        out[f'{nm}_mean']=s.mean(); out[f'{nm}_darkfrac']=dark.mean()
        gq=np.percentile(g,[50,90,99,99.9]); 
        for i,v in zip(['50','90','99','999'],gq): out[f'{nm}_g{i}']=v
        bm=g>gq[2]; out[f'{nm}_bd_I']=s[g>gq[1]].mean(); out[f'{nm}_bstep']=s[~dark].mean()-s[dark].mean()
        out[f'{nm}_otsu_sep']=(s[~dark].mean()-s[dark].mean())/(s.std()+1e-6); out[f'{nm}_sharp']=gq[3]/(gq[0]+1e-6)
        lap=filters.laplace(s); out[f'{nm}_lapvar']=lap.var()
    return out
rows=[]
for sp in ['train','test']:
    fs=sorted(os.listdir(f'{R}/{sp}'))
    res=Parallel(n_jobs=2)(delayed(f)(f'{R}/{sp}/{x}') for x in fs)
    for x,r in zip(fs,res): r['ID']=x[:-4]; rows.append(r)
pd.DataFrame(rows).set_index('ID').to_csv('ctr_feats.csv')
