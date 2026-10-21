# Handcrafted per-image features (no dataset-level statistics; every number depends only on the image itself).
# grain_*: identical to runs/grain/feat_grain.py ; raw_*: identical to runs/raw/feat_raw.py
# hand_*/tex_*: original generator code was not found -> re-implemented per-image with the same column semantics.
import numpy as np,pandas as pd,cv2,sys
from multiprocessing import Pool
from skimage import filters,segmentation,measure,morphology,restoration,feature
from common import *
def grain(im8):
    im=im8.astype(np.float32)/255
    sig=restoration.estimate_sigma(im)
    den=cv2.bilateralFilter(im,9,0.08,5); den=cv2.GaussianBlur(den,(0,0),1.0)
    med=np.median(den); s=np.std(den)+1e-6; z=(den-med)/s
    pore=z<-2.5; pore=morphology.remove_small_objects(pore,8)
    ridge=filters.sato(den,sigmas=[1,2],black_ridges=True); rthr=ridge>np.percentile(ridge,80)
    interior=~(rthr|pore); dist=cv2.distanceTransform(interior.astype(np.uint8),cv2.DIST_L2,3)
    mk=measure.label(morphology.h_maxima(dist,2)); lab=segmentation.watershed(ridge,mk,mask=~pore)
    props=measure.regionprops(lab,intensity_image=den)
    A=np.array([p.area for p in props]); I=np.array([p.intensity_mean for p in props])
    ecc=np.array([p.major_axis_length/(p.minor_axis_length+1e-3) for p in props]); ori=np.array([p.orientation for p in props])
    big=A>30; A,I,ecc,ori=A[big],I[big],ecc[big],ori[big]
    Iz=(I-np.average(I,weights=A))/s; mat=np.percentile(I,70)
    d={'sig':sig,'blur':cv2.Laplacian(im,cv2.CV_32F).var(),'den_std':s,'ng':len(A),'A_mean':A.mean(),'A_med':np.median(A),'A_cv':A.std()/A.mean(),
       'inv_sqrt_d':1/np.sqrt(np.sqrt(A.mean())),'ecc_mean':np.average(ecc,weights=A),'ecc_med':np.median(ecc)}
    c2=np.average(np.cos(2*ori),weights=A); s2=np.average(np.sin(2*ori),weights=A); d['ori_align']=np.hypot(c2,s2)
    d['grain_shade_std']=np.sqrt(np.cov(I,aweights=A))/(sig+1e-3); d['grain_shade_std_rel']=np.sqrt(np.cov(I,aweights=A))/s
    for t in [0.5,1,1.5,2]:
        m=(mat-I)/(np.std(I)+1e-6)>t; d[f'ph_area{t}']=A[m].sum()/A.sum(); d[f'ph_n{t}']=m.mean()
    th=filters.threshold_otsu(I); dk=I<th
    d['otsu_frac']=A[dk].sum()/A.sum(); d['otsu_gap']=(I[~dk].mean()-I[dk].mean())/(sig+1e-3); d['otsu_gap_rel']=(I[~dk].mean()-I[dk].mean())/s
    d['dark_grain_A']=A[dk].mean() if dk.any() else 0
    d['pore_frac']=pore.mean(); d['pore_n']=measure.label(pore).max(); d['ridge_mean']=ridge.mean()/(s+1e-6)
    for q in [5,25,50,75,95]: d[f'Iq{q}']=np.percentile(Iz,q)
    return {'g_'+k:v for k,v in d.items()}
def raw(im8):
    im=im8.astype(np.float32); d={}
    for s in [0,1,2,4,8]:
        b=cv2.GaussianBlur(im,(0,0),s) if s else im
        d[f'mean{s}']=b.mean(); d[f'std{s}']=b.std()
        for q in [1,5,25,50,75,95,99]: d[f'q{q}_{s}']=np.percentile(b,q)
        gx=cv2.Sobel(b,cv2.CV_32F,1,0);gy=cv2.Sobel(b,cv2.CV_32F,0,1);gm=np.hypot(gx,gy); d[f'g{s}']=gm.mean(); d[f'g95_{s}']=np.percentile(gm,95)
    d['noise']=np.std(im-cv2.GaussianBlur(im,(0,0),1))
    return {'r_'+k:v for k,v in d.items()}
def corr_len(ac,ang):
    c=ac.shape[0]//2; v=[]
    for r in range(1,40):
        v.append(ac[int(round(c+r*np.sin(ang))),int(round(c+r*np.cos(ang)))])
    v=np.array(v); i=np.where(v<np.exp(-1))[0]; return float(i[0]+1) if len(i) else 40.
def hand(im8):
    im=im8.astype(np.float32)/255; b=cv2.GaussianBlur(im,(0,0),1.0); z=(b-np.median(b))/(b.std()+1e-6); d={}
    for t in [0.5,1.0,1.5,2.0]: d[f'dark{t}']=(z<-t).mean()
    d['pore']=(z<-2.5).mean()
    lab=measure.label(z<-1.0); A=np.bincount(lab.ravel())[1:]; A=A[A>=5]
    d['ncc']=len(A); d['ccmean']=A.mean() if len(A) else 0; d['cc200']=A[A>200].sum()/z.size if len(A) else 0
    gx=cv2.Sobel(b,cv2.CV_32F,1,0);gy=cv2.Sobel(b,cv2.CV_32F,0,1)
    Jxx,Jyy,Jxy=[cv2.GaussianBlur(v,(0,0),4) for v in (gx*gx,gy*gy,gx*gy)]
    d['coh']=float(np.mean(np.sqrt((Jxx-Jyy)**2+4*Jxy**2)/(Jxx+Jyy+1e-9)))
    th=0.5*np.arctan2(2*Jxy.mean(),Jxx.mean()-Jyy.mean())  # dominant gradient angle
    f=np.fft.fft2(z); ac=np.real(np.fft.ifft2(np.abs(f)**2)); ac=np.fft.fftshift(ac)/ac.max()
    d['x_across']=corr_len(ac,th); d['x_along']=corr_len(ac,th+np.pi/2); d['x_ratio']=d['x_along']/d['x_across']
    d['sharp']=float(np.std(cv2.Laplacian(im,cv2.CV_32F))/(im.std()+1e-6))
    for q in [1,5,10,25,75,90,95,99]: d[f'zq{q}']=np.percentile(z,q)
    return {'h_'+k:v for k,v in d.items()}
def tex(im8):
    d={}; q=(im8//8).astype(np.uint8)
    G=feature.graycomatrix(q,[1,2,4,8,16],[0,np.pi/4,np.pi/2,3*np.pi/4],levels=32,symmetric=True,normed=True)
    for p in ['contrast','homogeneity','energy','correlation','dissimilarity']:
        v=feature.graycoprops(G,p).mean(1)
        for j,dd in enumerate([1,2,4,8,16]): d[f'glcm_{p}_{dd}']=v[j]
    for P,Rr in [(8,1),(16,2),(24,3)]:
        l=feature.local_binary_pattern(im8,P,Rr,'uniform'); h=np.bincount(l.astype(int).ravel(),minlength=P+2)/l.size
        for j,v in enumerate(h): d[f'lbp{P}_{j}']=v
    im=im8.astype(np.float32)/255
    for fr in [0.05,0.1,0.2,0.3]:
        e=[np.abs(filters.gabor(im,fr,theta=t)[0]+1j*filters.gabor(im,fr,theta=t)[1]).mean() for t in np.arange(4)*np.pi/4]
        d[f'gab{fr}_m']=np.mean(e); d[f'gab{fr}_aniso']=(max(e)-min(e))/(np.mean(e)+1e-9)
    F=np.abs(np.fft.fftshift(np.fft.fft2(im-im.mean())))**2; c=np.array(F.shape)//2
    yy,xx=np.indices(F.shape); r=np.hypot(yy-c[0],xx-c[1]); edges=np.logspace(0,np.log10(c[0]),17)
    tot=F[(r>=1)&(r<c[0])].sum()
    for j in range(16): d[f'fft{j}']=F[(r>=edges[j])&(r<edges[j+1])].sum()/tot
    return {'t_'+k:v for k,v in d.items()}
def one(p):
    im8=cv2.imread(p,0); d={}
    for fn in (hand,tex,grain,raw): d.update(fn(im8))
    return d
if __name__=='__main__':
    for sp,ids in [('train',train_ids()),('test',test_ids())]:
        with Pool(8) as pl: rows=pl.map(one,[f'{R}/{sp}/{i}.png' for i in ids],chunksize=4)
        df=pd.DataFrame(rows).fillna(0); df.insert(0,'ID',ids); df.to_csv(f'{O}/feat/hand_{sp}.csv',index=False); print(sp,df.shape,flush=True)
