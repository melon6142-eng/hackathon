import os,sys;os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'./runs/dark_mask')
import numpy as np,pandas as pd,cv2,glob,itertools as it
from skimage import measure
from multiprocessing import Pool
from feat import mf  # dark_mask 특징 함수 그대로
O='./open'
# 단계(기존 규약, σ2 블러 z): 1=z<-1, 2=-1~-0.5, 3=-0.5~0.5, 4=>=0.5
ED=[-99,-1,-0.5,0.5,99]
def zz(im,G=None):
    g=cv2.GaussianBlur(im,(0,0),2);return (g-np.median(g))/(g.std()+1e-6) if G is None else (g-G[0])/G[1]
def stages(z):return [(z>=ED[k])&(z<ED[k+1]) for k in range(4)]
def cnt(m):
    lab=measure.label(m,connectivity=2)
    if lab.max()==0:return 0
    return int((np.bincount(lab.ravel())[1:]>=4).sum())
def ratio_feats(z,t,d):
    S=stages(z);n=[cnt(s) for s in S];a=[s.mean() for s in S]
    for k in range(4):d[f'{t}n{k+1}']=n[k];d[f'{t}a{k+1}']=a[k]
    grp=[(i,) for i in range(4)]+[c for c in it.combinations(range(4),2)]
    for A in grp:
        for B in grp:
            if set(A)&set(B) or A>=B and len(A)==len(B)==1 and A>B:continue
            if len(A)>1 and len(B)>1 and A>B:continue
            na=sum(n[i] for i in A);nb=sum(n[i] for i in B);aa=sum(a[i] for i in A);ab=sum(a[i] for i in B)
            nm=''.join(str(i+1) for i in A)+'_'+''.join(str(i+1) for i in B)
            d[f'{t}nr{nm}']=np.log((na+1)/(nb+1));d[f'{t}ar{nm}']=np.log((aa+1e-4)/(ab+1e-4))
def shape_feats(im,d):
    # hand/grain류 간단 형상·밝기 특징
    g=cv2.GaussianBlur(im,(0,0),1);d['mean']=im.mean();d['std']=im.std()
    for q in [1,5,25,50,75,95,99]:d[f'q{q}']=np.percentile(im,q)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0);gy=cv2.Sobel(g,cv2.CV_32F,0,1);gm=np.hypot(gx,gy)
    d['gm_mean']=gm.mean();d['gm_q99']=np.percentile(gm,99);d['gm_q90']=np.percentile(gm,90)
    e=cv2.Canny(np.clip(im,0,255).astype(np.uint8),40,90)>0;d['edge']=e.mean()
    th=cv2.threshold(np.clip(g,0,255).astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]>0
    lab=measure.label(th,connectivity=1);A=np.bincount(lab.ravel())[1:];A=A[A>=9]
    d['gr_n']=len(A);d['gr_Amean']=A.mean() if len(A) else 0;d['gr_Acv']=A.std()/A.mean() if len(A) else 0;d['otsu_frac']=th.mean()
def feats(im,G,full=True):
    d={};shape_feats(im,d);z=zz(im);ratio_feats(z,'i',d)
    if full:ratio_feats(zz(im,G),'g',d)
    s=stages(z);mf(s[0],'S1_',d);mf(s[1],'S2_',d);mf(s[0]|s[1],'S12_',d)
    return d
def ablate(im,rm,mode):
    z=zz(im);S=stages(z);out=im.copy();med=np.median(im)
    for k in rm:
        if mode=='zero':out[S[k]]=0
        elif mode=='med':out[S[k]]=med
        else: # 이웃 단계(가운데 쪽, 지워지지 않은 가장 가까운) 중앙값
            for nb in ([k+1,k+2,k+3] if k<2 else [k-1,k-2,k-3]):
                if 0<=nb<4 and nb not in rm and S[nb].any():out[S[k]]=np.median(im[S[nb]]);break
    return out
V=[('base',(),'')]+[(f'rm{k+1}_{m}',(k,),m) for k in range(4) for m in ['nb','zero']]+[(f'rm{a+1}{b+1}_nb',(a,b),'nb') for a,b in it.combinations(range(4),2)]
def one(a):
    p,G,mode=a;im=cv2.imread(p,0).astype(np.float32);i=os.path.basename(p)[:-4]
    if mode=='ratio':d=feats(im,G);d['ID']=i;return d
    R={}
    for nm,rm,m in V:
        d=feats(ablate(im,rm,m) if rm else im,G,full=False);d['ID']=i;d['V']=nm;R[nm]=d
    return list(R.values())
if __name__=='__main__':
    P=sorted(glob.glob(f'{O}/train/*.png'));Pt=sorted(glob.glob(f'{O}/test/*.png'))
    G=(146.5,15.3)
    with Pool(2) as pool:
        R=pool.map(one,[(p,G,'ratio') for p in P+Pt],chunksize=10);pd.DataFrame(R).to_csv('ratio_feats.csv',index=False);print('ratio done',flush=True)
        R=pool.map(one,[(p,G,'abl') for p in P],chunksize=5);pd.DataFrame([x for r in R for x in r]).to_csv('abl_feats.csv',index=False);print('abl done',flush=True)
