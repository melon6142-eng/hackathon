# usage: python eval3.py out.md  (reads dino_ft2/<tag>/oof.npy + dino_ft/<tag>)
import numpy as np,pandas as pd,os,sys,glob
from scipy.optimize import nnls
from sklearn.model_selection import KFold
D='./runs';y=pd.read_csv('./open/train.csv')['hardness'].values
b=np.load(f'{D}/oof_blend1.npy');bt=pd.read_csv('./out/submission_blend1_cv12.95.csv').iloc[:,1].values.copy()
M={}
for base in [f'{D}/dino_ft',f'{D}/dino_ft2']:
    for p in glob.glob(f'{base}/*/oof.npy'):
        t=os.path.basename(os.path.dirname(p))
        if t.startswith('dino') or t=='cnx_full_w': M[t]=(np.load(p),np.load(p.replace('oof','test')))
r=lambda p:np.sqrt(((p-y)**2).mean());hb=lambda p:(p-y)[y>230].mean()
def ncv(cols,reps=20):
    X=np.column_stack(cols+[np.ones(500)]);o=[]
    for s in range(reps):
        p=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X): w,_=nnls(X[tr],y[tr]);p[va]=X[va]@w
        o.append(r(p))
    return np.mean(o),np.std(o)
def avg(ts): return (np.mean([M[t][0] for t in ts],0),np.mean([M[t][1] for t in ts],0))
L=[f'BL1 OOF {r(b):.3f} hi {hb(b):.1f}','','| 모델 | OOF | hi편향 | BL1+모델 nCV |','|---|---|---|---|']
grp={'B_orig_avg':[t for t in ['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2'] if t in M],
     'B_trans_avg':[t for t in ['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2'] if t in M]}
for k,v in grp.items():
    if len(v)>1: M[k]=avg(v)
for t in sorted(M):
    o=M[t][0];m,s=ncv([b,o]);L.append(f'| {t} | {r(o):.3f} | {hb(o):.1f} | {m:.3f}±{s:.3f} |')
cands={'BL1+B1seed(orig,trans)':['dinoB_orig','dinoB_trans'],
       'BL1+Bavg(orig,trans)':['B_orig_avg','B_trans_avg']}
for u in ['dinoB_trans_u6','dinoB_trans_u8']:
    if u in M: cands[f'BL1+Bavg+{u}']=['B_orig_avg','B_trans_avg',u]
cands['BL1+all']=[t for t in M if t not in ('B_orig_avg','B_trans_avg')]
L+=['','| 블렌드 | nCV(20회) | 가중치 |','|---|---|---|'];best=None
for k,ts in cands.items():
    ts=[t for t in ts if t in M]
    m,s=ncv([b]+[M[t][0] for t in ts]);X=np.column_stack([b]+[M[t][0] for t in ts]+[np.ones(500)]);w=nnls(X,y)[0]
    L.append(f'| {k} | {m:.3f}±{s:.3f} | {dict(zip(["BL1"]+ts+["c"],w.round(3)))} |')
    if best is None or m<best[0]: best=(m,k,ts,w)
m,k,ts,w=best;pt=np.column_stack([bt]+[M[t][1] for t in ts]+[np.ones(len(bt))])@w
L+=['',f'BEST {k} nCV {m:.3f}  test mean {pt.mean():.2f} (BL1 {bt.mean():.2f})']
if m<12.62:
    sub=pd.read_csv('./out/submission_blend1_cv12.95.csv');sub.iloc[:,1]=pt
    fn=f'{D}/dino_ft2/cand_{k.replace("+","_").replace("(","").replace(")","").replace(",","-")}_cv{m:.2f}.csv';sub.to_csv(fn,index=False);L.append(f'CSV {fn}')
open(sys.argv[1],'w').write('\n'.join(L));print('\n'.join(L))
