import numpy as np,pandas as pd,glob,os
from skimage import io,filters,measure
from scipy import ndimage as ndi
from joblib import Parallel,delayed
fs=sorted(glob.glob('./open/train/*.png'))+sorted(glob.glob('./open/test/*.png'))
def f(p):
    im=io.imread(p).astype(float);im=im if im.ndim==2 else im[...,0]
    z=(im-im.mean())/(im.std()+1e-6);d={'ID':os.path.basename(p)[:-4]}
    gy,gx=np.gradient(ndi.gaussian_filter(im,1))
    for s in [2,4,8,16]:
        J11=ndi.gaussian_filter(gx*gx,s);J22=ndi.gaussian_filter(gy*gy,s);J12=ndi.gaussian_filter(gx*gy,s)
        l=np.sqrt((J11-J22)**2+4*J12**2);c=l/(J11+J22+1e-9)
        d[f'st_coh{s}_mean']=c.mean();d[f'st_coh{s}_q90']=np.quantile(c,.9)
        th=0.5*np.arctan2(2*J12,J11-J22);w=(J11+J22)*c
        R=np.abs((w*np.exp(2j*th)).sum())/(w.sum()+1e-9);d[f'st_globR{s}']=R
    th=np.arctan2(gy,gx);m=np.hypot(gx,gy);h,_=np.histogram(np.mod(th,np.pi),18,weights=m);h=h/h.sum()
    d['go_max']=h.max();d['go_ent']=-(h*np.log(h+1e-12)).sum();d['go_R']=np.abs((m*np.exp(2j*th)).sum())/m.sum()
    for t in [-1.5,-1,-0.5]:
        b=ndi.gaussian_filter(z,1)<t;lb=measure.label(b);
        if lb.max()==0: d[f'cc{t}_max']=0;d[f'cc{t}_top3']=0;d[f'cc{t}_maxar']=1;continue
        rp=measure.regionprops(lb);A=np.sort([r.area for r in rp])[::-1];big=max(rp,key=lambda r:r.area)
        d[f'cc{t}_max']=A[0]/b.size;d[f'cc{t}_top3']=A[:3].sum()/b.size;d[f'cc{t}_maxar']=big.major_axis_length/(big.minor_axis_length+1)
    return d
pd.DataFrame(Parallel(8)(delayed(f)(p) for p in fs)).to_csv('feat3.csv',index=False)
