# Per-image unsupervised structure measurements on TRAIN images only (no labels, no test). Output aux_train.npy (500,K)
import numpy as np,torch,cv2
X,_=torch.load('~/AI/daker/cnn/imgs.pt');G=(X[:,0].numpy()*255).astype(np.uint8)
pore_thr=np.percentile(G[::5].ravel(),1.0)  # fixed global dark threshold (label-free)
N=256;fy=np.fft.fftfreq(N)[:,None];fx=np.fft.fftfreq(N)[None];fr=np.sqrt(fx**2+fy**2);wl=np.where(fr>0,1/np.maximum(fr,1e-9),1e9)
bands=[(0,4),(4,16),(16,64),(64,1e9)]
def bandE(g):
    P=np.abs(np.fft.fft2(g-g.mean()))**2;tot=P.sum()+1e-9
    return [P[(wl>=a)&(wl<b)].sum()/tot for a,b in bands]
def feat(g):
    gf=g.astype(np.float32);gb=cv2.GaussianBlur(g,(0,0),1.5)
    # dark phase (Otsu)
    t,_=cv2.threshold(gb,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU);dk=(gb<t).astype(np.uint8)
    n,lab,st,_=cv2.connectedComponentsWithStats(dk,8);ar=st[1:,4] if n>1 else np.array([0])
    dfrac=dk.mean();dconn=ar.max()/max(dk.sum(),1);dlogn=np.log1p(n-1)
    # pores
    pm=(gb<=pore_thr).astype(np.uint8);pn,_,ps,_=cv2.connectedComponentsWithStats(pm,8);pa=ps[1:,4]
    pa=pa[pa>=4];pcount=np.log1p(len(pa));psize=np.log1p(pa.mean()) if len(pa) else 0.
    # boundaries / grain size
    e=cv2.Canny(gb,30,90)>0;edens=e.mean()
    dt=cv2.distanceTransform((~e).astype(np.uint8),cv2.DIST_L2,3);gsz=np.log1p(dt[dt>0].mean())
    loc=dt[cv2.dilate((dt>=cv2.dilate(dt,np.ones((7,7)))).astype(np.uint8)*(dt>1),None)>0] if True else None
    r=dt[(dt==cv2.dilate(dt,np.ones((9,9))))&(dt>1)];gspread=r.std()/(r.mean()+1e-6) if len(r)>3 else 0.
    # orientation anisotropy (structure tensor coherence)
    gx=cv2.Sobel(gf,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gf,cv2.CV_32F,0,1,ksize=3)
    J=np.array([(gx*gx).mean(),(gy*gy).mean(),(gx*gy).mean()]);coh=np.sqrt((J[0]-J[1])**2+4*J[2]**2)/(J[0]+J[1]+1e-9)
    # FFT bands + unevenness (tile std of 16-64 band and edge density)
    be=np.log(np.array(bandE(gf))+1e-9)
    T=[gf[i:i+64,j:j+64] for i in range(0,256,64) for j in range(0,256,64)]
    tb=[];te=[]
    for k,tt in enumerate(T):
        P=np.abs(np.fft.fft2(tt-tt.mean()))**2;f=np.sqrt(np.fft.fftfreq(64)[:,None]**2+np.fft.fftfreq(64)[None]**2);w=np.where(f>0,1/np.maximum(f,1e-9),1e9)
        tb.append(P[(w>=16)&(w<64)].sum());i,j=divmod(k,4);te.append(e[i*64:i*64+64,j*64:j*64+64].mean())
    tb=np.array(tb);uneven_b=tb.std()/(tb.mean()+1e-9);uneven_e=np.std(te)/(np.mean(te)+1e-9)
    return [gsz,gspread,edens,coh,dfrac,dconn,dlogn,pcount,psize,*be,uneven_b,uneven_e]
names=['grain_size_logDT','grain_size_spread','gb_density','orient_coherence','dark_frac','dark_largest_conn','dark_logncomp','pore_logcount','pore_logsize','fftE_lt4','fftE_4_16','fftE_16_64','fftE_gt64','uneven_band16_64','uneven_edge']
A=np.array([feat(g) for g in G],dtype=np.float32);np.save('aux_train.npy',A);open('aux_names.txt','w').write('\n'.join(names))
import pandas as pd;y=pd.read_csv('~/AI/daker/open/train.csv').hardness.values
from scipy.stats import spearmanr
for n,c in zip(names,A.T): print(f'{n:22s} mean {c.mean():.3f} sd {c.std():.3f} rho_y {spearmanr(c,y)[0]:+.3f}')
