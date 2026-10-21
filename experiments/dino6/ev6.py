# usage: python ev6.py  (expects runs/dino6/<tag>/{oof,test,oof_rot,test_rot}.npy copied from Mac Studio cnn/runs/<tag>/)
import numpy as np,pandas as pd,os,sys
from scipy.optimize import nnls
from sklearn.model_selection import KFold
D='./runs';H=f'{D}/dino6';y=pd.read_csv('./open/train.csv')['hardness'].values
ld=lambda t,k:np.load(f'{D}/{t}/{k}.npy');av=lambda ts,k:np.mean([ld(t,k) for t in ts],0)
O=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','dino_ft2/dinoB_orig_s2'];T=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','dino_ft2/dinoB_trans_s2']
sub=pd.read_csv('./out/submission_blend1_cv12.95.csv')
B=np.column_stack([np.load(f'{D}/oof_blend1.npy'),av(O,'oof'),av(T,'oof')]);BT=np.column_stack([sub.iloc[:,1].values,av(O,'test'),av(T,'test')])
cand={}
for tag in ['d4_rank','d6_maskcoh','d6_maskaux']+[f'dinoB_mask_s{s}' for s in (42,1,2)]:
    for suf in ['','_rot']:
        p=f'{H}/{tag}/oof{suf}.npy'
        if os.path.exists(p): cand[tag+suf]=(np.load(p),np.load(f'{H}/{tag}/test{suf}.npy'))
r=lambda p,t:np.sqrt(((p-t)**2).mean());hi=y>230;lo=y<170
def ncv(X,R=10):
    A=np.column_stack([X,np.ones(500)]);out=[]
    for s in range(R):
        pp=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(A): pp[va]=A[va]@nnls(A[tr],y[tr])[0]
        out.append((r(pp,y),(pp-y)[hi].mean(),(pp-y)[lo].mean()))
    return np.array(out).mean(0)
base=ncv(B);L=['| 구성 | 단독 OOF | nCV(10회) | Δ | hi | lo |','|---|---|---|---|---|---|',f'| base(BL1+Borig+Btrans) | - | {base[0]:.3f} | 0 | {base[1]:.1f} | {base[2]:.1f} |']
best=(base[0],None)
for k,(o,t) in cand.items():
    m=ncv(np.column_stack([B,o]));L.append(f'| +{k} | {r(o,y):.3f} | {m[0]:.3f} | {m[0]-base[0]:+.3f} | {m[1]:.1f} | {m[2]:.1f} |')
    if m[0]<best[0]: best=(m[0],k)
open(f'{H}/ev6.md','w').write('\n'.join(L));print('\n'.join(L))
if best[1] and base[0]-best[0]>=0.05:
    o,t=cand[best[1]];A=np.column_stack([B,o,np.ones(500)]);w=nnls(A,y)[0];pt=np.column_stack([BT,t,np.ones(len(t))])@w
    os.makedirs(f'{D}/round2',exist_ok=True);s=sub.copy();s.iloc[:,1]=pt;fn=f'{D}/round2/cand_base+{best[1]}_cv{best[0]:.2f}.csv';s.to_csv(fn,index=False);print('CAND',fn,w)
else: print('no candidate (improvement < 0.05)')
