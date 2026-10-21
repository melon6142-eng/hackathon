import os;os.environ['OMP_NUM_THREADS']='2'
import numpy as np,pandas as pd,sys
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
H='./runs/hi_rule'
F=pd.read_csv('dm_feats.csv').set_index('ID').replace([np.inf,-np.inf],np.nan);F=F.fillna(F.median())
def XX(sp):
    f=F.loc[F.index.str.startswith(sp.upper())].sort_index();o=pd.read_pickle(f'{H}/X_{sp}.pkl')
    return np.hstack([o.values,f.values]) if os.environ.get('WITHOLD') else f.values
y=pd.read_csv('./open/train.csv').hardness.values
Xv=XX('train');Xt=XX('test')
C=np.load(H+'/comp_oof.npy');Ct=np.load(H+'/comp_test.npy')
one=lambda M:np.column_stack([M,np.ones(len(M))])
r=lambda p:np.sqrt(((p-y)**2).mean())
def clf(kind):
    if kind=='L1':return make_pipeline(StandardScaler(),LogisticRegression(penalty='l1',C=.1,solver='liblinear',class_weight='balanced'))
    if kind=='L1w':return make_pipeline(StandardScaler(),LogisticRegression(penalty='l1',C=.05,solver='liblinear'))
    if kind=='tree3':return DecisionTreeClassifier(max_depth=3,min_samples_leaf=8,class_weight='balanced',random_state=0)
    if kind=='baseonly':return LogisticRegression(class_weight='balanced')
def Zf(kind,X,b):return b[:,None]/100 if kind=='baseonly' else np.column_stack([X,b])
def fit_predict(tr,Xa,Ca,Xb,Cb,kind,TH,TL,sides):
    ya=y[tr];w=nnls(one(Ca),ya)[0];ba=one(Ca)@w;bb=one(Cb)@w
    res=ya-ba;P_in=[];P_out=[]
    for side in sides:
        lab=(ya>TH) if side=='hi' else (ya<TL)
        pin=np.zeros(len(tr))
        for i,(a,v) in enumerate(KFold(5,shuffle=True,random_state=7).split(tr)):
            pin[v]=clf(kind).fit(Zf(kind,Xa[a],ba[a]),lab[a]).predict_proba(Zf(kind,Xa[v],ba[v]))[:,1]
        pout=clf(kind).fit(Zf(kind,Xa,ba),lab).predict_proba(Zf(kind,Xb,bb))[:,1]
        sg=1 if side=='hi' else -1;P_in.append(sg*pin);P_out.append(sg*pout)
    if not sides:return bb,bb
    co=nnls(np.column_stack(P_in),res)[0]
    return bb,bb+np.column_stack(P_out)@co,co
reps=int(sys.argv[1]) if len(sys.argv)>1 else 5
cfgs=[(k,TH,s) for k in ['L1','tree3'] for TH in [220,230] for s in [('hi',),('hi','lo')]]
L=['| 판별기 | 고경도 임계 | 대상 | nCV RMSE | 기준 | hi편향 | lo편향 |','|---|---|---|---|---|---|---|']
best=None
for k,TH,s in cfgs:
    R=[];B=[];HB=[];LB=[]
    for rp in range(reps):
        p=np.zeros(500);p0=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+rp).split(Xv):
            o=fit_predict(tr,Xv[tr],C[tr],Xv[va],C[va],k,TH,170,list(s));p0[va]=o[0];p[va]=o[1]
        R.append(r(p));B.append(r(p0));HB.append((p-y)[y>230].mean());LB.append((p-y)[y<170].mean())
    L.append(f'| {k} | {TH} | {"+".join(s)} | {np.mean(R):.3f}±{np.std(R):.3f} | {np.mean(B):.3f} | {np.mean(HB):.1f} | {np.mean(LB):.1f} |');print(L[-1],flush=True)
    if best is None or np.mean(R)<best[0]:best=(np.mean(R),k,TH,s,np.mean(B))
m,k,TH,s,b0=best
bb,pt,co=fit_predict(np.arange(500),Xv,C,Xt,Ct,k,TH,170,list(s))
L+=['',f'BEST {k} T{TH} {s} nCV {m:.3f} vs base {b0:.3f}; 전체적합 가산계수 {co.round(2)}; test 평균 {pt.mean():.2f} (base {bb.mean():.2f}), 보정량 |Δ|>5인 test {np.sum(np.abs(pt-bb)>5)}장']
if m<b0-0.03:
    sub=pd.read_csv('./out/submission_blend1_cv12.95.csv');sub.iloc[:,1]=pt
    fn=f'cand_dm{os.environ.get("WITHOLD","")}_{k}_T{TH}_{"".join(s)}_cv{m:.2f}.csv';sub.to_csv(fn,index=False);L.append('CSV '+os.path.abspath(fn))
else:L.append('개선 폭이 반복CV 오차 수준 이하라 CSV 미생성')
open(os.environ.get('OUT','corr.md'),'w').write('\n'.join(L));print('\n'.join(L))
