import os; os.environ['OMP_NUM_THREADS']='1'
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2, sys
from skimage import filters, segmentation, measure, morphology, restoration
from multiprocessing import Pool
R='./open'
def acf_aniso(b):
    b=b-b.mean(); F=np.fft.fft2(b); ac=np.real(np.fft.ifft2(np.abs(F)**2)); ac=np.fft.fftshift(ac)/ac.max()
    c=128; w=ac[c-40:c+41,c-40:c+41]; yy,xx=np.mgrid[-40:41,-40:41]
    m=w>0.6 ; # central lobe
    lab=measure.label(m); m=lab==lab[40,40]
    W=w*m; S=W.sum(); Ixx=(W*xx*xx).sum()/S; Iyy=(W*yy*yy).sum()/S; Ixy=(W*xx*yy).sum()/S
    ev,evec=np.linalg.eigh(np.array([[Ixx,Ixy],[Ixy,Iyy]]))
    ang=np.degrees(np.arctan2(evec[1,1],evec[0,1]))%180
    return np.sqrt(ev[1]/ev[0]), ang, np.sqrt(ev[0]*ev[1]), m.sum()
def f(args):
    split,i=args
    im=cv2.imread(f'{R}/{split}/{i}.png',0).astype(np.float32)
    d={'ID':i}
    d['mean']=im.mean(); d['std']=im.std()
    sig=restoration.estimate_sigma(im/255)*255; d['noise_wav']=sig
    # local residual noise: im - median3 in flat areas
    den=cv2.fastNlMeansDenoising(im.astype(np.uint8),None,h=max(3,sig*1.2),templateWindowSize=5,searchWindowSize=15).astype(np.float32)
    res=im-den; d['noise_res']=res.std()
    # blur: gradient magnitude of denoised, high-pct
    g=np.hypot(cv2.Sobel(den,cv2.CV_32F,1,0),cv2.Sobel(den,cv2.CV_32F,0,1))
    d['grad99']=np.percentile(g,99); d['grad90']=np.percentile(g,90)
    # boundaries: dark thin ridges
    rid=filters.sato(den,sigmas=[1,1.5],black_ridges=True)
    thr=filters.threshold_otsu(rid); bd=rid>thr
    bd=morphology.remove_small_objects(bd,15)
    sk=morphology.skeletonize(bd)
    d['LA']=sk.mean()  # boundary length per area (px/px^2)
    ar,ang,geo,_=acf_aniso(cv2.GaussianBlur(sk.astype(np.float32),(0,0),3)); d['acf_ar']=ar; d['acf_ang']=ang
    # structure-tensor anisotropy on den
    gx=cv2.Sobel(den,cv2.CV_32F,1,0);gy=cv2.Sobel(den,cv2.CV_32F,0,1)
    J=np.array([[(gx*gx).mean(),(gx*gy).mean()],[(gx*gy).mean(),(gy*gy).mean()]]);ev=np.linalg.eigvalsh(J)
    d['st_ar']=np.sqrt(ev[1]/ev[0]); d['st_ang']=np.degrees(0.5*np.arctan2(2*J[0,1],J[0,0]-J[1,1]))%180
    # pores: very dark compact blobs
    med=np.median(den); mad=np.median(np.abs(den-med))+1e-3
    pz=(den-med)/mad
    pore=morphology.remove_small_objects(morphology.binary_opening(pz<-6,morphology.disk(1)),10)
    pl=measure.label(pore); pp=measure.regionprops(pl,intensity_image=den)
    pa=np.array([p.area for p in pp]) if pp else np.zeros(1)
    d['pore_n']=len(pp); d['pore_frac']=pore.mean(); d['pore_r']=np.sqrt(pa.mean()/np.pi) if len(pp) else 0
    d['pore_ecc']=np.mean([p.axis_major_length/(p.axis_minor_length+1e-3) for p in pp]) if pp else 1
    d['pore_I']=np.mean([p.intensity_mean for p in pp])-med if pp else 0
    # grains via watershed
    interior=~(bd|pore); dist=cv2.distanceTransform(interior.astype(np.uint8),cv2.DIST_L2,3)
    mk=measure.label(morphology.h_maxima(dist,1.5))
    lab=segmentation.watershed(rid,mk,mask=~pore)
    props=measure.regionprops(lab,intensity_image=den)
    A=np.array([p.area for p in props]);I=np.array([p.intensity_median for p in props])
    keep=A>20;A=A[keep];I=I[keep]
    d['ng']=len(A); d['A_mean']=A.mean(); d['A_med']=np.median(A)
    # phase levels: kmeans 2 on cell intensity
    from sklearn.cluster import KMeans
    km=KMeans(2,n_init=5,random_state=0).fit(I.reshape(-1,1)); c=km.cluster_centers_.ravel(); lo=np.argmin(c)
    dark=km.labels_==lo
    d['ph_contrast']=c.max()-c.min(); d['ph_cnt']=dark.mean(); d['ph_area']=A[dark].sum()/A.sum()
    d['ph_bright']=c.max(); d['ph_dark']=c.min()
    d['cell_Istd_bright']=I[~dark].std(); d['cell_Istd_dark']=I[dark].std() if dark.sum()>1 else 0
    d['A_dark']=A[dark].mean() if dark.any() else 0; d['A_bright']=A[~dark].mean()
    d['bd_I']=den[sk].mean()-np.median(den[~bd])
    return d
if __name__=='__main__':
    jobs=[]
    for s in ['train','test']:
        ids=sorted(x[:-4] for x in os.listdir(f'{R}/{s}'))
        jobs+= [(s,i) for i in ids]
    if len(sys.argv)>1: jobs=jobs[:int(sys.argv[1])]
    with Pool(3) as p: out=p.map(f,jobs,chunksize=10)
    df=pd.DataFrame(out); df.to_csv('gen_feats.csv',index=False); print(df.describe().T)
