# 새 특징: 기공(P) 군, 결정립 크기 불균질(G) 군. train 500장만, 이미지별 독립.
import os,sys; os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'~/AI/daker/phase_sep')
import numpy as np,pandas as pd,cv2
from skimage import measure
from scipy.spatial import cKDTree
from multiprocessing import Pool
from seg import segment,P0
O='~/AI/daker/open'
def cv(a): a=np.asarray(a,float); return float(a.std()/(a.mean()+1e-9)) if len(a) else 0.
def f1(i):
    im=cv2.imread(f'{O}/train/{i}.png',0); S=segment(im,P0); pore=S['pore']; H,W=im.shape; d={'ID':i}
    rp=measure.regionprops(measure.label(pore)); A=np.array([r.area for r in rp],float); C=np.array([r.centroid for r in rp]).reshape(-1,2)
    n=len(A); d['P_frac']=pore.mean(); d['P_n']=n; d['P_logn']=np.log1p(n)
    d['P_Amean']=A.mean() if n else 0; d['P_Amed']=np.median(A) if n else 0; d['P_Aq90']=np.percentile(A,90) if n else 0
    d['P_Acv']=cv(A) if n>1 else 0; d['P_Amax']=A.max() if n else 0; d['P_eqd']=np.sqrt(4*d['P_Amean']/np.pi)
    # 공간 군집: Clark-Evans R, 타일 개수 분산/평균 비(VMR)
    if n>=3:
        dd,_=cKDTree(C).query(C,2); obs=dd[:,1].mean(); exp=0.5/np.sqrt(n/(H*W)); d['P_CE']=obs/exp
        d['P_nn_cv']=cv(dd[:,1])
    else: d['P_CE']=1.0; d['P_nn_cv']=0.
    for t in (32,64):
        g=np.zeros((H//t,W//t))
        for y,x in C: g[min(int(y)//t,H//t-1),min(int(x)//t,W//t-1)]+=1
        m=g.mean(); d[f'P_vmr{t}']=g.var()/m if m>0 else 1.0
        fr=pore.reshape(H//t,t,W//t,t).mean((1,3)); d[f'P_fsd{t}']=fr.std()
    # 결정립 크기 불균질: 타일별 평균 결정립 면적(중심 기준)과 입계 밀도의 CV
    lab=S['lab']; ok=S['ok']; GA=S['A']
    pr=measure.regionprops_table(lab,properties=('label','centroid')); idx=pr['label']-1; k=ok[idx]
    cy=pr['centroid-0'][k]; cx=pr['centroid-1'][k]; ga=GA[idx][k]; sk=S['skel'].astype(float)
    for t in (16,32,64):
        bd=sk.reshape(H//t,t,W//t,t).mean((1,3)); d[f'G_bcv{t}']=cv(bd.ravel())  # 입계 밀도 ∝ 1/결정립 크기
        d[f'G_lbsd{t}']=float(np.log(bd.ravel()+1e-3).std())
        if t>=32:
            ty=(cy//t).astype(int); tx=(cx//t).astype(int); key=ty*(W//t)+tx
            m=[np.log(ga[key==q]).mean() for q in np.unique(key) if (key==q).sum()>=2]
            d[f'G_lAsd{t}']=float(np.std(m)) if len(m)>1 else 0.
    d['G_Acv']=cv(ga); d['G_lAsd']=float(np.log(ga).std())
    return d
if __name__=='__main__':
    ids=pd.read_csv(f'{O}/train.csv').ID.tolist()
    with Pool(10) as p: R=p.map(f1,ids)
    pd.DataFrame(R).to_csv('feat_new.csv',index=False); print('done',len(R))
