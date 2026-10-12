import os; os.environ['OMP_NUM_THREADS']='1'
import warnings; warnings.filterwarnings("ignore")
import numpy as np,pandas as pd,cv2
from skimage import measure
from scipy import ndimage as ndi
from multiprocessing import Pool
R='./open'
def comp(m,d,tag):
    lab,n=ndi.label(m); 
    if n==0:
        for k in ['max','wmean','top','perc']: d[f'{tag}_{k}']=0
        d[f'{tag}_area']=0;d[f'{tag}_perim']=0;d[f'{tag}_euler']=0;return
    s=np.bincount(lab.ravel())[1:]; A=m.sum()
    d[f'{tag}_max']=s.max()/m.size; d[f'{tag}_wmean']=(s**2).sum()/A/m.size; d[f'{tag}_top']=s.max()/A
    tb=set(lab[0][lab[0]>0])&set(lab[-1][lab[-1]>0]); lr=set(lab[:,0][lab[:,0]>0])&set(lab[:,-1][lab[:,-1]>0])
    d[f'{tag}_perc']=int(len(tb)>0 or len(lr)>0)
    d[f'{tag}_area']=A/m.size; d[f'{tag}_perim']=(m[1:]!=m[:-1]).sum()+(m[:,1:]!=m[:,:-1]).sum()
    d[f'{tag}_euler']=measure.euler_number(m,connectivity=2)
def f(a):
    s,i=a; im=cv2.imread(f'{R}/{s}/{i}.png',0).astype(np.float32); d={'ID':i}
    sm=cv2.GaussianBlur(im,(0,0),2)
    z=(sm-sm.mean())/(sm.std()+1e-6)
    for t in (96,112,128,144): comp(sm<t,d,f'abs{t}')
    for t in (-2,-1.5,-1,-0.5,0): comp(z<t,d,f'z{t}')
    # orientation coherence multi-scale
    gx=cv2.Sobel(cv2.GaussianBlur(im,(0,0),1),cv2.CV_32F,1,0);gy=cv2.Sobel(cv2.GaussianBlur(im,(0,0),1),cv2.CV_32F,0,1)
    for r in (2,4,8,16):
        Jxx=cv2.GaussianBlur(gx*gx,(0,0),r);Jyy=cv2.GaussianBlur(gy*gy,(0,0),r);Jxy=cv2.GaussianBlur(gx*gy,(0,0),r)
        coh=np.sqrt((Jxx-Jyy)**2+4*Jxy**2)/(Jxx+Jyy+1e-6); d[f'coh{r}_mean']=coh.mean(); d[f'coh{r}_q90']=np.quantile(coh,.9)
        if r==8:
            th=0.5*np.arctan2(2*Jxy,Jxx-Jyy); c2=np.cos(2*th);s2=np.sin(2*th)
            d['ori_global']=np.hypot(c2.mean(),s2.mean())
            # orientation correlation length
            def acl(x):
                x=x-x.mean();F=np.fft.fft2(x);ac=np.real(np.fft.ifft2(np.abs(F)**2));return ac/ac[0,0]
            ac=(acl(c2)+acl(s2))/2; rr=np.hypot(*np.meshgrid(np.fft.fftfreq(256)*256,np.fft.fftfreq(256)*256))
            prof=np.array([ac[(rr>=k)&(rr<k+1)].mean() for k in range(64)]); k=np.where(prof<np.exp(-1))[0]
            d['ori_corrlen']=k[0] if len(k) else 64
            # domain size: quantize orientation into 6 bins, components
            q=((th+np.pi/2)/(np.pi/6)).astype(int)%6; sz=[]
            for b in range(6):
                lab,n=ndi.label(q==b); 
                if n: sz+=list(np.bincount(lab.ravel())[1:])
            sz=np.array(sz); d['ori_dom_wmean']=(sz**2).sum()/sz.sum(); d['ori_dom_max']=sz.max()
    # high-frequency energy ratios
    P=np.abs(np.fft.fftshift(np.fft.fft2(im-im.mean())))**2; rr=np.hypot(*np.mgrid[-128:128,-128:128]); tot=P.sum()
    for lo,hi in [(4,8),(8,16),(16,32),(32,64),(64,128)]: d[f'hf_{lo}_{hi}']=P[(rr>=lo)&(rr<hi)].sum()/tot
    L=[np.var(cv2.GaussianBlur(im,(0,0),s1)-cv2.GaussianBlur(im,(0,0),2*s1)) for s1 in (1,2,4,8)]
    for k,v in zip((1,2,4,8),L): d[f'dog{k}_rat']=v/(sum(L)+1e-6)
    return d
if __name__=='__main__':
    jobs=[(s,x[:-4]) for s in ['train','test'] for x in sorted(os.listdir(f'{R}/{s}'))]
    with Pool(2) as p: out=p.map(f,jobs,chunksize=10)
    pd.DataFrame(out).to_csv('morph_feats.csv',index=False);print('done')
