# usage: python evf.py feats.npz NAME [sets...]  — frozen feature Ridge, strict nested 10x5 with base blend
import numpy as np,pandas as pd,sys,os,warnings;warnings.filterwarnings('ignore')
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
D='./runs';y=pd.read_csv('./open/train.csv')['hardness'].values
ld=lambda t,k:np.load(f'{D}/{t}/{k}.npy');av=lambda ts,k:np.mean([ld(t,k) for t in ts],0)
O=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','dino_ft2/dinoB_orig_s2'];T=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','dino_ft2/dinoB_trans_s2']
sub=pd.read_csv('./out/submission_blend1_cv12.95.csv')
B=np.column_stack([np.load(f'{D}/oof_blend1.npy'),av(O,'oof'),av(T,'oof')]);BT=np.column_stack([sub.iloc[:,1].values,av(O,'test'),av(T,'test')])
f=np.load(sys.argv[1]);name=sys.argv[2]
def get(keys):
    M=[]
    for k in keys:
        a=f[k]
        if k.startswith('cov'): a=PCA(256,random_state=0).fit_transform(StandardScaler().fit_transform(a))  # unsupervised train+test
        M.append(a)
    return np.concatenate(M,1)
SETS={'gem':['gem_L0','gem_L1'],'cov':['cov_L0','cov_L1'],'clsmean':['cls_L1','mean_L1'],'gem+cov':['gem_L0','gem_L1','cov_L0','cov_L1'],'all':['cls_L1','mean_L1','gem_L0','gem_L1','cov_L0','cov_L1']}
if len(sys.argv)>3: SETS={k:SETS[k] for k in sys.argv[3:]}
mk=lambda:make_pipeline(StandardScaler(),RidgeCV(np.logspace(0,5,21)))
r=lambda p,t:np.sqrt(((p-t)**2).mean());hi=y>230;lo=y<170
def ncv(X,R=10):
    A=np.column_stack([X,np.ones(500)]);out=[]
    for s in range(R):
        pp=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(A): pp[va]=A[va]@nnls(A[tr],y[tr])[0]
        out.append((r(pp,y),(pp-y)[hi].mean(),(pp-y)[lo].mean()))
    return np.array(out).mean(0)
base=ncv(B);os.makedirs(f'{D}/overnight/oof',exist_ok=True);L=[]
for sn,keys in SETS.items():
    Xa=get(keys);X,XT=Xa[:500],Xa[500:]
    # solo OOF (folds 42) + test pred
    oof=np.zeros(500)
    for tr,va in KFold(5,shuffle=True,random_state=42).split(X): oof[va]=mk().fit(X[tr],y[tr]).predict(X[va])
    te=mk().fit(X,y).predict(XT);np.save(f'{D}/overnight/oof/{name}_{sn}_oof.npy',oof);np.save(f'{D}/overnight/oof/{name}_{sn}_test.npy',te)
    # strict nested: ridge refit inside each outer fold, inner OOF for NNLS
    res=[]
    for s in range(10):
        pp=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X):
            io=np.zeros(len(tr))
            for i,j in KFold(5,shuffle=True,random_state=7).split(tr): io[j]=mk().fit(X[tr[i]],y[tr[i]]).predict(X[tr[j]])
            full=mk().fit(X[tr],y[tr]).predict(X[va])
            w=nnls(np.column_stack([B[tr],io,np.ones(len(tr))]),y[tr])[0];pp[va]=np.column_stack([B[va],full,np.ones(len(va))])@w
        res.append((r(pp,y),(pp-y)[hi].mean(),(pp-y)[lo].mean()))
    m=np.array(res).mean(0)
    A=np.column_stack([B,oof,np.ones(500)]);w=nnls(A,y)[0]
    cr=[np.corrcoef(oof,B[:,i])[0,1] for i in range(3)]
    line=f'| {name}_{sn} | {r(oof,y):.3f} | {(oof-y)[hi].mean():.1f} | {(oof-y)[lo].mean():.1f} | {m[0]:.3f} | {m[0]-base[0]:+.3f} | {m[1]:.1f} | {cr[0]:.2f}/{cr[1]:.2f}/{cr[2]:.2f} | {w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}/{w[3]:.2f} |'
    print(line,flush=True);L.append(line)
    if m[0]<=12.55:
        pt=np.column_stack([BT,te,np.ones(len(te))])@w;s_=sub.copy();s_.iloc[:,1]=pt;fn=f'{D}/overnight/cand_base+{name}_{sn}_cv{m[0]:.3f}.csv';s_.to_csv(fn,index=False);print('CAND',fn)
open(f'{D}/overnight/evf_{name}.md','w').write('\n'.join(L)+'\n')
print('base',base)
