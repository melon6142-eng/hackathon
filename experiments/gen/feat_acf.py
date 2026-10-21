import os; os.environ['OMP_NUM_THREADS']='1'
import warnings; warnings.filterwarnings("ignore")
import numpy as np,pandas as pd,cv2
from skimage import restoration
from multiprocessing import Pool
R='./open'
def L(prof):
    k=np.where(prof<np.exp(-1))[0]; return k[0] if len(k) else len(prof)
def f(a):
    s,i=a; im=cv2.imread(f'{R}/{s}/{i}.png',0)
    sig=restoration.estimate_sigma(im/255)*255
    den=cv2.fastNlMeansDenoising(im,None,h=max(3,sig*1.2),templateWindowSize=5,searchWindowSize=15).astype(np.float32)
    med=np.median(den); mad=np.median(np.abs(den-med))+1e-3; pore=((den-med)/mad<-6).astype(np.uint8)
    inp=cv2.inpaint(den.astype(np.uint8),cv2.dilate(pore,np.ones((5,5))),3,cv2.INPAINT_TELEA).astype(np.float32)
    d={'ID':i,'sig_raw':sig}
    # raw-image ACF: lag-1 correlation of noise -> blur of noise? noise white? 
    r=im.astype(np.float32)-cv2.GaussianBlur(im.astype(np.float32),(0,0),3)
    d['noise_lag1']=np.corrcoef(r[:,1:].ravel(),r[:,:-1].ravel())[0,1]
    for nm,x in [('den',inp-inp.mean())]:
        F=np.fft.fft2(x*np.outer(np.hanning(256),np.hanning(256))); ac=np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F)**2))); ac/=ac.max()
        c=128; yy,xx=np.mgrid[-30:31,-30:31]; w=ac[c-30:c+31,c-30:c+31]
        Wm=np.clip(w-np.exp(-1),0,None); S=Wm.sum()
        M=np.array([[(Wm*xx*xx).sum(),(Wm*xx*yy).sum()],[(Wm*xx*yy).sum(),(Wm*yy*yy).sum()]])/S
        ev,evec=np.linalg.eigh(M); d['acf_major']=np.sqrt(ev[1]); d['acf_minor']=np.sqrt(ev[0]); d['acf_ratio']=np.sqrt(ev[1]/ev[0])
        d['acf_angle']=np.degrees(np.arctan2(evec[1,1],evec[0,1]))%180; d['acf_area']=np.sqrt(ev[0]*ev[1])
        th=np.radians(d['acf_angle']); 
        for nm2,t in [('Lmaj',th),('Lmin',th+np.pi/2)]:
            rr=np.arange(0,60); pr=ac[np.clip((c+rr*np.sin(t)).round().astype(int),0,255),np.clip((c+rr*np.cos(t)).round().astype(int),0,255)]
            d[nm2]=L(pr); d[nm2+'_ac10']=pr[10]
        d['var_low']=np.var(cv2.GaussianBlur(inp,(0,0),4)); d['var_cell']=np.var(inp)
    d['pore_frac2']=pore.mean()
    return d
if __name__=='__main__':
    jobs=[(s,x[:-4]) for s in ['train','test'] for x in sorted(os.listdir(f'{R}/{s}'))]
    with Pool(3) as p: out=p.map(f,jobs,chunksize=10)
    pd.DataFrame(out).to_csv('acf_feats.csv',index=False);print('done')
