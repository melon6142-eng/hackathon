# (A) 밝기 단계별 배치·형상 특징 (이미지별 독립, 회전 불변). usage: python featA.py
import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,cv2
from skimage import measure
from scipy.spatial import cKDTree
from multiprocessing import Pool
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../open')
BINS={'vlight':(1,9),'light':(0.5,1),'L0':(-0.5,0.5),'L1':(-1,-0.5),'L2':(-1.5,-1),'L3':(-2.5,-1.5),'pore':(-99,-2.5),
      'cum1':(-2.5,-0.5),'cum2':(-2.5,-1)}
K3=np.ones((3,3),np.uint8)
def mfeat(m,p):
    d={}; n=m.size; d[p+'frac']=m.mean()
    t=m.reshape(8,32,8,32).mean((1,3)); d[p+'tile_sd']=t.std(); d[p+'tile_cv']=t.std()/(t.mean()+1e-6)
    lab=measure.label(m,connectivity=2)
    if lab.max()==0:
        for k in ['ncc','A_mean','A_med','A_q90','A_max','A_cv','ar','sol','pa','align','nn_mean','nn_cv','CE','K15']: d[p+k]=0
        return d
    pr=measure.regionprops_table(lab,properties=('area','axis_major_length','axis_minor_length','solidity','perimeter','orientation','centroid'))
    A=pr['area'].astype(float); k=A>=4
    if k.sum()==0: k=A>=1
    A=A[k]; d[p+'ncc']=k.sum()
    d[p+'A_mean']=A.mean(); d[p+'A_med']=np.median(A); d[p+'A_q90']=np.percentile(A,90); d[p+'A_max']=A.max(); d[p+'A_cv']=A.std()/A.mean()
    w=A/A.sum()
    d[p+'ar']=(w*pr['axis_major_length'][k]/(pr['axis_minor_length'][k]+1)).sum()
    d[p+'sol']=(w*pr['solidity'][k]).sum(); d[p+'pa']=pr['perimeter'][k].sum()/A.sum()
    o=pr['orientation'][k]; d[p+'align']=np.hypot((w*np.cos(2*o)).sum(),(w*np.sin(2*o)).sum())
    c=np.c_[pr['centroid-0'][k],pr['centroid-1'][k]]
    if len(c)>2:
        tr=cKDTree(c); dd,_=tr.query(c,2); nn=dd[:,1]; lam=len(c)/n
        d[p+'nn_mean']=nn.mean(); d[p+'nn_cv']=nn.std()/(nn.mean()+1e-6); d[p+'CE']=nn.mean()/(0.5/np.sqrt(lam))
        pairs=tr.query_pairs(15,output_type='ndarray'); d[p+'K15']=(2*len(pairs)/len(c))/lam/(np.pi*225)
    else: d[p+'nn_mean']=d[p+'nn_cv']=d[p+'CE']=d[p+'K15']=0
    return d
def f(path):
    im=cv2.imread(path,0).astype(np.float32); g=cv2.GaussianBlur(im,(0,0),2); z=(g-np.median(g))/(g.std()+1e-6)
    d={}; M={}
    for nm,(lo,hi) in BINS.items():
        M[nm]=(z>=lo)&(z<hi); d.update(mfeat(M[nm],nm+'_'))
    # 단계 간 인접성: A 경계(팽창 링)가 B에 닿는 비율
    for a,b in [('pore','L2'),('pore','L3'),('pore','L0'),('L2','L0'),('L2','L1'),('L3','L2'),('cum2','L0'),('L2','light')]:
        ring=cv2.dilate(M[a].astype(np.uint8),K3).astype(bool)&~M[a]
        d[f'adj_{a}_{b}']=(ring&M[b]).sum()/(ring.sum()+1)
    return d
def run(sp):
    ids=sorted(x[:-4] for x in os.listdir(f'{R}/{sp}'))
    with Pool(2) as p: rows=p.map(f,[f'{R}/{sp}/{i}.png' for i in ids],chunksize=20)
    df=pd.DataFrame(rows).fillna(0); df.insert(0,'ID',ids); return df
if __name__=='__main__':
    for sp in ['train','test']: run(sp).to_csv(f'A_{sp}.csv',index=False); print(sp,'done',flush=True)
