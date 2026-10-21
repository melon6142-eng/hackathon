import os;os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,cv2,glob
from skimage import measure
from multiprocessing import Pool
O='./open'
ZE=np.arange(-3,3.01,.25);ZM=[-2,-1.5,-1,-.5,0,.5,1,1.5,2]  # 연결성분용 주요 z구간 경계
ATH=[64,80,96,112,128,144,160,176,192]
def cc(m):
    lab=measure.label(m,connectivity=2);A=np.bincount(lab.ravel())[1:];A=A[A>=4]
    return len(A),(A.mean() if len(A) else 0),(A.max() if len(A) else 0)
def one(p):
    im=cv2.imread(p,0).astype(np.float32);g=cv2.GaussianBlur(im,(0,0),2);d={'ID':os.path.basename(p)[:-4]}
    z=(g-g.mean())/(g.std()+1e-6)
    h=np.histogram(np.clip(z,-3.0,2.999),ZE)[0]/z.size
    for i,v in enumerate(h):d[f'R_z{ZE[i]:+.2f}']=v
    for q in [1,5,10,25,50,75,90,95,99]:d[f'R_zq{q}']=np.percentile(z,q);d[f'A_q{q}']=np.percentile(im,q)
    E=[-99]+ZM+[99]
    for i in range(len(E)-1):
        n,m,mx=cc((z>=E[i])&(z<E[i+1]));t=f'R_cc[{E[i]},{E[i+1]})';d[t+'_n']=n;d[t+'_mA']=m;d[t+'_maxA']=mx
    for k,B in [(16,16),(32,8)]:
        hh=np.histogram(im,np.arange(0,257,B))[0]/im.size
        for i,v in enumerate(hh):d[f'A_h{k}_{i*B}']=v
    for t in ATH:
        m=g<t;d[f'A_lt{t}']=m.mean();n,mA,mx=cc(m);d[f'A_cc<{t}_n']=n;d[f'A_cc<{t}_mA']=mA;d[f'A_cc<{t}_maxA']=mx
    d['A_mean']=im.mean();d['A_std']=im.std()
    return d
if __name__=='__main__':
    P=sorted(glob.glob(f'{O}/train/*.png'))+sorted(glob.glob(f'{O}/test/*.png'))
    with Pool(6) as pl:pd.DataFrame(pl.map(one,P,chunksize=10)).to_csv('bf_feats.csv',index=False)
