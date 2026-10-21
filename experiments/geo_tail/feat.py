import numpy as np,pandas as pd,sys
from skimage import io,filters,segmentation,feature,measure
from scipy import ndimage as ndi
from joblib import Parallel,delayed
def f(p):
    im=io.imread(p,as_gray=True).astype(float)
    if im.max()>1.5: im/=255
    F=np.abs(np.fft.fftshift(np.fft.fft2(im-im.mean())))**2
    h,w=F.shape;yy,xx=np.indices(F.shape);r=np.hypot(yy-h//2,xx-w//2).astype(int)
    ps=np.bincount(r.ravel(),F.ravel())/np.maximum(np.bincount(r.ravel()),1)
    k=np.arange(2,min(h,w)//4);sl=np.polyfit(np.log(k),np.log(ps[k]+1e-12),1)[0]
    lo=ps[2:8].sum()/ps[2:min(h,w)//2].sum()
    sm=filters.gaussian(im,1.5);g=filters.sobel(sm)
    mk=feature.peak_local_max(sm,min_distance=4,labels=None,exclude_border=False)
    m=np.zeros(im.shape,int);m[tuple(mk.T)]=np.arange(1,len(mk)+1)
    lab=segmentation.watershed(g,m);A=np.bincount(lab.ravel())[1:];A=A[A>0].astype(float)
    q=np.quantile(A,np.linspace(.1,.99,10))
    d=dict(psd_slope=sl,psd_lowfrac=lo,ws_n=len(A),ws_Amean=A.mean(),ws_Acv=A.std()/A.mean(),ws_top5=np.sort(A)[-5:].mean(),ws_top10frac=np.sort(A)[-max(1,len(A)//10):].sum()/A.sum(),ws_skew=((A-A.mean())**3).mean()/A.std()**3)
    for i,v in enumerate(q): d[f'ws_q{i}']=v
    return d
ids=pd.read_csv('./open/train.csv')['ID']
R=Parallel(8)(delayed(f)(f'./open/train/{i}.png') for i in ids)
pd.DataFrame(R).assign(ID=ids).to_csv('geo_train.csv',index=False)
