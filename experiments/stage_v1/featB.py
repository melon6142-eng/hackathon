# (B) 패턴 거리 비교 특징. 기준 집합은 반드시 fit()에 들어온 학습 fold에서만 고름.
import numpy as np,pandas as pd,cv2,os
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
R='./open'; H=os.path.dirname(os.path.abspath(__file__))
def build_reps():
    fn=f'{H}/reps.npz'
    if os.path.exists(fn): z=np.load(fn); return z['cdf'],z['tex'],z['dino']
    cdfs=[]
    for sp in ['train','test']:
        for i in sorted(x[:-4] for x in os.listdir(f'{R}/{sp}')):
            im=cv2.imread(f'{R}/{sp}/{i}.png',0).astype(np.float32); g=cv2.GaussianBlur(im,(0,0),2); z=(g-np.median(g))/(g.std()+1e-6)
            h,_=np.histogram(np.clip(z,-4,4),bins=64,range=(-4,4)); cdfs.append(np.cumsum(h)/h.sum())
    tex=np.vstack([pd.read_csv(f'{R}/combo_v1/feat/tex_{sp}.csv').drop(columns='ID').values for sp in ['train','test']]).astype(np.float32)
    D=768; sel=[(L,k) for L in range(4) for k in (1,2)]
    dino=np.vstack([np.hstack([np.load(f'{H}/../dino/vitb14_252_{sp}.npy',mmap_mode='r')[:,(L*4+k)*D:(L*4+k+1)*D] for L,k in sel]) for sp in ['train','test']]).astype(np.float32)
    np.savez(fn,cdf=np.array(cdfs,np.float32),tex=tex,dino=dino); return np.array(cdfs,np.float32),tex,dino
CDF,TEX,DINO=build_reps()   # rows: 0..499 train, 500..1499 test
def dmats(qi,ri,st):
    """distance matrices query x ref for each representation (st: fitted scalers from ref)"""
    out={}
    out['wass']=np.abs(CDF[qi][:,None,:]-CDF[ri][None,:,:]).sum(-1)*(8/64)
    tq=(TEX[qi]-st['tm'])/st['ts']; tr=(TEX[ri]-st['tm'])/st['ts']
    out['tex']=np.sqrt(((tq[:,None]-tr[None])**2).sum(-1))
    pq=st['pca'].transform(DINO[qi]); pr=st['pca'].transform(DINO[ri])
    a=pq/np.linalg.norm(pq,axis=1,keepdims=True); b=pr/np.linalg.norm(pr,axis=1,keepdims=True); out['cos']=1-a@b.T
    wq=pq/st['pv']; wr=pr/st['pv']; out['maha']=np.sqrt(((wq[:,None]-wr[None])**2).sum(-1))
    return out
def feats(qi,ri,ry,st):
    D=dmats(qi,ri,st); o=np.argsort(ry); F=[]; names=[]
    qs=np.quantile(ry,[0,.2,.4,.6,.8,1])
    for nm,M in D.items():
        for k in (5,10,20):
            lo=M[:,o[:k]].mean(1); hi=M[:,o[-k:]].mean(1); F+= [lo,hi,lo-hi]; names+=[f'{nm}_lo{k}',f'{nm}_hi{k}',f'{nm}_lmh{k}']
        for b in range(5):
            m=(ry>=qs[b])&(ry<=qs[b+1]); F.append(M[:,m].mean(1)); names.append(f'{nm}_q{b}')
        for k in (5,10,20):
            idx=np.argsort(M,1)[:,:k]; w=1/(np.take_along_axis(M,idx,1)+1e-6); F.append((ry[idx]*w).sum(1)/w.sum(1)); names.append(f'{nm}_knn{k}')
    return np.column_stack(F),names
def fitstats(ri):
    p=PCA(64,random_state=0).fit(DINO[ri]); return {'tm':TEX[ri].mean(0),'ts':TEX[ri].std(0)+1e-6,'pca':p,'pv':np.sqrt(p.explained_variance_)}
class PatternDist:
    """X = row index column. fit(): train rows get inner-OOF features (self never in ref); transform(other): ref=whole fit set."""
    def fit(s,X,y):
        s.ri=X[:,0].astype(int); s.y=np.asarray(y); s.st=fitstats(s.ri); s.F=np.zeros((len(s.ri),0)); parts={}
        for t,v in KFold(5,shuffle=True,random_state=7).split(s.ri):
            f,s.names=feats(s.ri[v],s.ri[t],s.y[t],fitstats(s.ri[t])); parts.update({j:r for j,r in zip(v,f)})
        s.F=np.array([parts[j] for j in range(len(s.ri))]); return s
    def transform(s,X):
        qi=X[:,0].astype(int)
        if len(qi)==len(s.ri) and (qi==s.ri).all(): return s.F
        return feats(qi,s.ri,s.y,s.st)[0]
