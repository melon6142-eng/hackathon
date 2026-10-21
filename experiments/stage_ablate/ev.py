import os;os.environ['OMP_NUM_THREADS']='2'
import numpy as np,pandas as pd,warnings,sys;warnings.filterwarnings('ignore')
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.linear_model import LogisticRegression,RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
H='./runs/hi_rule'
y=pd.read_csv('./open/train.csv').hardness.values
C=np.load(f'{H}/comp_oof.npy');one=lambda M:np.column_stack([M,np.ones(len(M))])
base=one(C)@nnls(one(C),y)[0];r=lambda p:np.sqrt(((p-y)**2).mean())
zone=(base>=195)&(base<=235);lab=(y>230).astype(int);idx=np.where(zone)[0]
def clean(X):X=X.replace([np.inf,-np.inf],np.nan);return X.fillna(X.median()).fillna(0).values
def zauc(X,R=5):
    A=[]
    for s in range(R):
        p=np.zeros(len(idx))
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=s).split(idx,lab[idx]):
            m=make_pipeline(StandardScaler(),LogisticRegression(penalty='l1',C=.1,solver='liblinear',class_weight='balanced')).fit(X[idx[tr]],lab[idx[tr]]);p[va]=m.predict_proba(X[idx[va]])[:,1]
        A.append(auc(lab[idx],p))
    return np.mean(A),np.std(A)
def hcv(X,R=5):  # 특징만으로 경도 회귀 (alpha는 내부 CV → nested)
    o=[]
    for s in range(R):
        p=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X):
            p[va]=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,5,25))).fit(X[tr],y[tr]).predict(X[va])
        o.append(r(p))
    return np.mean(o),np.std(o)
def rcv(X,R=5):  # 최고 블렌드 잔차 보정 nCV
    o=[]
    for s in range(R):
        p=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X):
            w=nnls(one(C[tr]),y[tr])[0];bt=one(C[tr])@w;bv=one(C[va])@w
            p[va]=bv+make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(0,5,20))).fit(X[tr],y[tr]-bt).predict(X[va])
        o.append(r(p))
    return np.mean(o),np.std(o)
L=[f'기준 블렌드 nCV(동일 프로토콜): {np.mean([r(base)]):.3f}  구간 n={zone.sum()}, 고경도 {lab[zone].sum()}']
if sys.argv[1]=='ratio':
    F=pd.read_csv('ratio_feats.csv').set_index('ID').sort_index();F=F.loc[F.index.str.startswith('TRAIN')]
    u=pd.Series({c:auc(lab[zone],F[c].fillna(0).values[zone]) for c in F.columns});uu=(u-.5).abs().sort_values(ascending=False)
    L.append('\n[구간 195~235 단변량 AUC top20 (in-sample; 0.5에서 먼 순)]');L+=[f'- {c}: {u[c]:.3f}' for c in uu.index[:20]]
    rr=[c for c in F.columns if c[1:3] in('nr','ar') or (c[1]in'na' and c[2:].isdigit())]
    sets={'개수비(nr) 이미지별':F.filter(regex='^inr'),'개수비(nr) 전역':F.filter(regex='^gnr'),'개수비 전체':F.filter(regex='^.nr'),'면적비(ar) 전체':F.filter(regex='^.ar'),
          '단계 개수+면적(원값)':F.filter(regex='^.[na][1-4]$'),'비율특징 전체':F[rr]}
    L.append('\n| 특징셋 | 개수 | 구간 hi AUC (nested 5×5) | 경도 단독 nCV | 블렌드 보정 nCV |\n|---|---|---|---|---|')
    for k,X in sets.items():
        X=clean(X);a=zauc(X);h=hcv(X);c=rcv(X)
        L.append(f'| {k} | {X.shape[1]} | {a[0]:.3f}±{a[1]:.3f} | {h[0]:.2f} | {c[0]:.3f}±{c[1]:.3f} |');print(L[-1],flush=True)
    open('ev_ratio.md','w').write('\n'.join(L))
else:
    F=pd.read_csv('abl_feats.csv');res=[]
    for v,g in F.groupby('V',sort=False):
        g=g.set_index('ID').sort_index().drop(columns='V')
        for fs,X in [('형상',g.loc[:,:'otsu_frac']),('개수비',g.filter(regex='^i[na]')),('dark_mask',g.filter(regex='^S')),('전체',g)]:
            X=clean(X);a=zauc(X,3);h=hcv(X,3);c=rcv(X,3);res.append(dict(V=v,set=fs,auc=a[0],hrmse=h[0],resid=c[0]));print(res[-1],flush=True)
    D=pd.DataFrame(res);D.to_csv('abl_res.csv',index=False)
