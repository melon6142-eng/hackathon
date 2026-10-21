# grain-level features; per-image independent preprocessing only. usage: python feat_grain.py
import numpy as np, pandas as pd, cv2, os, sys
from skimage import filters, segmentation, measure, morphology, restoration
from multiprocessing import Pool
R=os.path.join(os.path.dirname(__file__),'../../open')
def f(path):
    im=cv2.imread(path,0).astype(np.float32)/255
    sig=restoration.estimate_sigma(im)
    den=cv2.bilateralFilter(im,9,0.08,5); den=cv2.GaussianBlur(den,(0,0),1.0)
    med=np.median(den); s=np.std(den)+1e-6; z=(den-med)/s
    pore=z<-2.5; pore=morphology.remove_small_objects(pore,8)
    ridge=filters.sato(den,sigmas=[1,2],black_ridges=True)
    rthr=ridge>np.percentile(ridge,80)
    # markers: grain interiors
    interior=~(rthr|pore); dist=cv2.distanceTransform(interior.astype(np.uint8),cv2.DIST_L2,3)
    mk=measure.label(morphology.h_maxima(dist,2))
    lab=segmentation.watershed(ridge,mk,mask=~pore)
    props=measure.regionprops(lab,intensity_image=den)
    A=np.array([p.area for p in props]); I=np.array([p.intensity_mean for p in props])
    ecc=np.array([p.major_axis_length/(p.minor_axis_length+1e-3) for p in props])
    ori=np.array([p.orientation for p in props]); MA=np.array([p.axis_major_length for p in props]); MI=np.array([p.axis_minor_length for p in props])
    big=A>30; A,I,ecc,ori,MA,MI=A[big],I[big],ecc[big],ori[big],MA[big],MI[big]
    Iz=(I-np.average(I,weights=A))/s
    # matrix = brightest mode via area-weighted median of upper part
    mat=np.percentile(I,70); Ir=(I-mat)/(np.std(I)+1e-6)
    d={}
    d['sig']=sig; d['blur']=cv2.Laplacian(im,cv2.CV_32F).var(); d['den_std']=s
    d['ng']=len(A); d['A_mean']=A.mean(); d['A_med']=np.median(A); d['A_cv']=A.std()/A.mean()
    d['inv_sqrt_d']=1/np.sqrt(np.sqrt(A.mean()))
    d['ecc_mean']=np.average(ecc,weights=A); d['ecc_med']=np.median(ecc)
    c2=np.average(np.cos(2*ori),weights=A); s2=np.average(np.sin(2*ori),weights=A); d['ori_align']=np.hypot(c2,s2)
    d['grain_shade_std']=np.sqrt(np.cov(I,aweights=A))/(sig+1e-3)
    d['grain_shade_std_rel']=np.sqrt(np.cov(I,aweights=A))/s
    for t in [0.5,1,1.5,2]:
        m=(mat-I)/(np.std(I)+1e-6)>t; d[f'ph_area{t}']=A[m].sum()/A.sum(); d[f'ph_n{t}']=m.mean()
    # otsu on grain shades -> two-phase split
    th=filters.threshold_otsu(I); dk=I<th
    d['otsu_frac']=A[dk].sum()/A.sum(); d['otsu_gap']=(I[~dk].mean()-I[dk].mean())/(sig+1e-3)
    d['otsu_gap_rel']=(I[~dk].mean()-I[dk].mean())/s
    d['dark_grain_A']=A[dk].mean() if dk.any() else 0
    d['MA_w']=np.average(MA,weights=A); d['MI_w']=np.average(MI,weights=A); d['MA_med']=np.median(MA); d['MI_med']=np.median(MI); d['MA_q90']=np.percentile(MA,90); d['A_w']=np.average(A,weights=A); d['dkMA']=np.average(MA[dk],weights=A[dk]) if dk.any() else 0; d['dkMI']=np.average(MI[dk],weights=A[dk]) if dk.any() else 0
    d['pore_frac']=pore.mean(); d['pore_n']=measure.label(pore).max()
    d['bnd_density']=rthr.mean(); d['ridge_mean']=ridge.mean()/(s+1e-6)
    for q in [5,25,50,75,95]: d[f'Iq{q}']=np.percentile(Iz,q)
    return d
def run(split):
    ids=sorted(x[:-4] for x in os.listdir(f'{R}/{split}'))
    with Pool(2) as p: rows=p.map(f,[f'{R}/{split}/{i}.png' for i in ids])
    df=pd.DataFrame(rows); df.insert(0,'ID',ids); return df
if __name__=='__main__':
    out=os.path.dirname(__file__)
    for sp in ['train','test']: run(sp).to_csv(f'{out}/grain2_{sp}.csv',index=False); print(sp,'done',flush=True)
