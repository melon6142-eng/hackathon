# Nested CV eval for d11_multilayer, d11_cropaug vs base BL1+Borig3+Btrans3 (12.601)
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
D='./runs'; H=f'{D}/overnight'
y=pd.read_csv('./open/train.csv')['hardness'].values
sub=pd.read_csv('./out/submission_blend1_cv12.95.csv')
ld=lambda t,k: np.load(f'{D}/{t}/{k}.npy'); av=lambda ts,k: np.mean([ld(t,k) for t in ts],0)
r=lambda p,t: np.sqrt(((p-t)**2).mean()); hi=y>230; lo=y<170
O3=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','dino_ft2/dinoB_orig_s2']
T3=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','dino_ft2/dinoB_trans_s2']
B=np.column_stack([np.load(f'{D}/oof_blend1.npy'), av(O3,'oof'), av(T3,'oof')])
BT=np.column_stack([sub.iloc[:,1].values, av(O3,'test'), av(T3,'test')])
NEW={k:(np.load(f'{H}/{k}/oof.npy'),np.load(f'{H}/{k}/test.npy')) for k in ['d11_multilayer','d11_cropaug','d8_dldl_orig','d10_canon']}
def A1(X): return np.column_stack([X,np.ones(len(X))])
def ncv(X,R=10):
    out=[]
    for s in range(R):
        pp=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X):
            pp[va]=A1(X[va])@nnls(A1(X[tr]),y[tr])[0]
        out.append((r(pp,y),(pp-y)[hi].mean(),(pp-y)[lo].mean()))
    return np.array(out).mean(0)
fw=lambda X: nnls(A1(X),y)[0]
base=ncv(B); print(f'BASE nCV={base[0]:.3f} hi={base[1]:.1f} lo={base[2]:.1f}')
L=['### Solo OOF','| tag | OOF | hi | lo | corr BL1/Bo/Bt |','|---|---|---|---|---|']
for k in ['d11_multilayer','d11_cropaug']:
    o=NEW[k][0]; c=[np.corrcoef(B[:,j],o)[0,1] for j in range(3)]
    s=f'| {k} | {r(o,y):.3f} | {(o-y)[hi].mean():.1f} | {(o-y)[lo].mean():.1f} | {c[0]:.3f}/{c[1]:.3f}/{c[2]:.3f} |'; print(s); L.append(s)
L+=['','### Nested CV (10×5, NNLS+intercept) vs base','| 구성 | nCV | Δ | hi | lo | full NNLS w |','|---|---|---|---|---|---|',
    f'| base | {base[0]:.3f} | 0 | {base[1]:.1f} | {base[2]:.1f} | {np.round(fw(B),3).tolist()} |']
rows=[]
def rep(lab,X,XT,sel=False):
    m=ncv(X); w=fw(X)
    s=f'| {lab} | {m[0]:.3f} | {m[0]-base[0]:+.3f} | {m[1]:.1f} | {m[2]:.1f} | {np.round(w,3).tolist()} |'
    print(s); L.append(s); rows.append((lab,m[0],m[1],X,XT,sel))
cols=lambda ks: (np.column_stack([B]+[NEW[k][0] for k in ks]),np.column_stack([BT]+[NEW[k][1] for k in ks]))
for cb in [['d11_multilayer'],['d11_cropaug'],['d11_multilayer','d11_cropaug'],['d11_multilayer','d8_dldl_orig']]:
    rep('+'+'+'.join(cb),*cols(cb))
def orig_avg(Bx,BxT,n0,ks):
    n=n0+len(ks)
    X=np.column_stack([Bx[:,0],(n0*Bx[:,1]+sum(NEW[k][0] for k in ks))/n,Bx[:,2]])
    XT=np.column_stack([BxT[:,0],(n0*BxT[:,1]+sum(NEW[k][1] for k in ks))/n,BxT[:,2]]); return X,XT
for ks in [['d11_multilayer'],['d11_cropaug'],['d11_multilayer','d11_cropaug'],['d11_multilayer','d8_dldl_orig','d10_canon'],['d11_multilayer','d11_cropaug','d8_dldl_orig','d10_canon']]:
    rep('Borig avg ← +'+'+'.join(ks),*orig_avg(B,BT,3,ks))
O_ds2=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','overnight/dinoB_orig_s3']; T_ds2=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','overnight/dinoB_trans_s3']
Bd=np.column_stack([B[:,0],av(O_ds2,'oof'),av(T_ds2,'oof')]); BdT=np.column_stack([BT[:,0],av(O_ds2,'test'),av(T_ds2,'test')])
for ks in [['d10_canon','d8_dldl_orig'],['d10_canon','d8_dldl_orig','d11_multilayer'],['d10_canon','d8_dldl_orig','d11_multilayer','d11_cropaug']]:
    rep('[sel-bias] drop_s2+s3, Borig ← +'+'+'.join(ks),*orig_avg(Bd,BdT,3,ks),sel=True)
best=min(rows,key=lambda z:z[1]); clean=min([z for z in rows if not z[5]],key=lambda z:z[1])
L+=['',f'BEST(all): {best[0]} nCV={best[1]:.3f} Δ{best[1]-base[0]:+.3f} hi={best[2]:.1f}',f'BEST(no post-hoc seed drop): {clean[0]} nCV={clean[1]:.3f} Δ{clean[1]-base[0]:+.3f} hi={clean[2]:.1f}']
print(L[-2]); print(L[-1])
for tag,z in [('best',best),('clean',clean)]:
    if z[1]<=base[0]-0.05:
        X,XT=z[3],z[4]; pt=A1(XT)@fw(X); s2=sub.copy(); s2.iloc[:,1]=pt
        fn=f'{H}/cand_d11_{tag}_cv{z[1]:.3f}.csv'; s2.to_csv(fn,index=False); L.append(f'CANDIDATE {fn}'); print('CAND',fn)
if best[1]>base[0]-0.05: L.append('no candidate (gain < 0.05)'); print('no candidate')
open(f'{H}/ev_d11.md','w').write('\n'.join(L)+'\n')
