import numpy as np,pandas as pd,os
from skimage import io,filters,segmentation,feature,measure
from scipy import ndimage as ndi
from joblib import Parallel,delayed
def gini(a):
    a=np.sort(a);n=len(a);return (2*np.arange(1,n+1)-n-1).dot(a)/(n*a.sum()+1e-12)
def f(p):
    im=io.imread(p,as_gray=True).astype(float)
    if im.max()>1.5: im/=255
    sm=filters.gaussian(im,1.5);g=filters.sobel(sm)
    mk=feature.peak_local_max(sm,min_distance=4,exclude_border=False)
    m=np.zeros(im.shape,int);m[tuple(mk.T)]=np.arange(1,len(mk)+1)
    lab=segmentation.watershed(g,m)
    P=measure.regionprops(lab,intensity_image=im)
    L=np.array([r.label for r in P]);A=np.array([r.area for r in P],float)
    ori=np.degrees(np.array([r.orientation for r in P]))
    ar=np.array([r.major_axis_length/max(r.minor_axis_length,1) for r in P]);I=np.array([r.mean_intensity for r in P])
    idx={l:i for i,l in enumerate(L)}
    a=np.concatenate([lab[:,:-1].ravel(),lab[:-1,:].ravel()]);b=np.concatenate([lab[:,1:].ravel(),lab[1:,:].ravel()])
    k=a!=b;pr=np.unique(np.sort(np.stack([a[k],b[k]],1),1),axis=0)
    i=np.array([idx[x] for x in pr[:,0]]);j=np.array([idx[x] for x in pr[:,1]])
    v=(A[i]>=15)&(A[j]>=15);i,j=i[v],j[v]
    da=np.abs(ori[i]-ori[j])%180;da=np.minimum(da,180-da)
    E={'dang':da,'larea':np.abs(np.log(A[i]/A[j])),'dar':np.abs(ar[i]-ar[j]),'dI':np.abs(I[i]-I[j])}
    d={}
    for n,x in E.items(): d[f'{n}_mean']=x.mean();d[f'{n}_med']=np.median(x);d[f'{n}_q90']=np.quantile(x,.9)
    d['same_dir15']=(da<15).mean()
    el=ar>1.3;d['same_dir15_elong']=(da[el[i]&el[j]]<15).mean() if (el[i]&el[j]).sum()>3 else np.nan
    d['A_cv']=A.std()/A.mean();d['A_gini']=gini(A);d['ar_cv']=ar.std()/ar.mean();d['ar_gini']=gini(ar-1);d['nbr_deg']=2*len(i)/len(A)
    for s in (4,8):
        Ixx,Ixy,Iyy=feature.structure_tensor(im,sigma=s,order='rc')
        coh=np.sqrt((Ixx-Iyy)**2+4*Ixy**2)/(Ixx+Iyy+1e-9)
        d[f'coh{s}_var']=coh.var();t=coh[:256//16*16,:256//16*16].reshape(16,16,16,16).mean((1,3));d[f'coh{s}_tilevar']=t.var();d[f'coh{s}_tilecv']=t.std()/t.mean()
        th=0.5*np.arctan2(2*Ixy,Ixx-Iyy);c2=np.cos(2*th)*coh;s2=np.sin(2*th)*coh
        c2t=c2[:256].reshape(16,16,16,16).mean((1,3));s2t=s2[:256].reshape(16,16,16,16).mean((1,3))
        d[f'ori{s}_tileagree']=np.hypot(c2t.mean(),s2t.mean())/(np.hypot(c2t,s2t).mean()+1e-9)
    return d
O='./open'
tr=pd.read_csv(f'{O}/train.csv').ID.tolist();te=sorted(x[:-4] for x in os.listdir(f'{O}/test'))
ps=[f'{O}/train/{i}.png' for i in tr]+[f'{O}/test/{i}.png' for i in te]
R=Parallel(os.cpu_count())(delayed(f)(p) for p in ps)
pd.DataFrame(R).assign(ID=tr+te).to_csv('ga_feats.csv',index=False);print('ok')
