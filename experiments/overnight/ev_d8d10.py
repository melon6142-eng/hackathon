# Nested CV eval for d8_dldl_orig, d9_cls_orig, d10_hiexp (incl gated mix), d10_canon vs base BL1+Borig3+Btrans3
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
B=np.column_stack([ld('oof_blend1','') if False else np.load(f'{D}/oof_blend1.npy'), av(O3,'oof'), av(T3,'oof')])
BT=np.column_stack([sub.iloc[:,1].values, av(O3,'test'), av(T3,'test')])
NEW={k:(np.load(f'{H}/{k}/oof.npy'),np.load(f'{H}/{k}/test.npy')) for k in ['d8_dldl_orig','d9_cls_orig','d10_hiexp','d10_canon']}
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
base=ncv(B); print(f'BASE nCV={base[0]:.3f} hi={base[1]:.1f} lo={base[2]:.1f} w={np.round(fw(B),3)}')
L=['### Solo OOF','| tag | OOF | hi | lo | corr BL1/Bo/Bt |','|---|---|---|---|---|']
for k,(o,_) in NEW.items():
    c=[np.corrcoef(B[:,j],o)[0,1] for j in range(3)]
    L.append(f'| {k} | {r(o,y):.3f} | {(o-y)[hi].mean():.1f} | {(o-y)[lo].mean():.1f} | {c[0]:.3f}/{c[1]:.3f}/{c[2]:.3f} |')
L+=['','### Nested CV (10×5, NNLS+intercept) vs base','| 구성 | nCV | Δ | hi | lo | maxcorr(new vs base) | full NNLS w (BL1,Bo,Bt,new...,b) |','|---|---|---|---|---|---|---|',
    f'| base | {base[0]:.3f} | 0 | {base[1]:.1f} | {base[2]:.1f} | - | {np.round(fw(B),3).tolist()} |']
rows=[]
def rep(lab,X,XT):
    m=ncv(X); w=fw(X); c=np.corrcoef(X.T); mc=float(np.max(np.abs(c[3:,:3]))) if X.shape[1]>3 else float('nan')
    s=f'| {lab} | {m[0]:.3f} | {m[0]-base[0]:+.3f} | {m[1]:.1f} | {m[2]:.1f} | {mc:.3f} | {np.round(w,3).tolist()} |'
    print(s); L.append(s); rows.append((lab,m[0],X,XT))
combos=[['d8_dldl_orig'],['d9_cls_orig'],['d10_hiexp'],['d10_canon'],['d8_dldl_orig','d9_cls_orig'],
        ['d8_dldl_orig','d10_canon'],['d9_cls_orig','d10_canon'],['d8_dldl_orig','d9_cls_orig','d10_canon'],
        ['d8_dldl_orig','d9_cls_orig','d10_hiexp','d10_canon']]
for cb in combos:
    rep('+'+'+'.join(cb),np.column_stack([B]+[NEW[k][0] for k in cb]),np.column_stack([BT]+[NEW[k][1] for k in cb]))
# cheap: fold new orig-channel heads into orig average (as extra members)
for k in ['d8_dldl_orig','d9_cls_orig','d10_canon']:
    X=np.column_stack([B[:,0],(3*B[:,1]+NEW[k][0])/4,B[:,2]]); XT=np.column_stack([BT[:,0],(3*BT[:,1]+NEW[k][1])/4,BT[:,2]])
    rep(f'Borig avg ← +{k} (4 members)',X,XT)
X=np.column_stack([B[:,0],(3*B[:,1]+NEW['d8_dldl_orig'][0]+NEW['d9_cls_orig'][0])/5,B[:,2]])
XT=np.column_stack([BT[:,0],(3*BT[:,1]+NEW['d8_dldl_orig'][1]+NEW['d9_cls_orig'][1])/5,BT[:,2]])
rep('Borig avg ← +d8+d9 (5 members)',X,XT)
# drop_s2 trick (+s3) then +best new
O_ds2=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','overnight/dinoB_orig_s3']; T_ds2=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','overnight/dinoB_trans_s3']
Bd=np.column_stack([B[:,0],av(O_ds2,'oof'),av(T_ds2,'oof')]); BdT=np.column_stack([BT[:,0],av(O_ds2,'test'),av(T_ds2,'test')])
rep('drop_s2+s3 (ref)',Bd,BdT)
for k in ['d8_dldl_orig','d9_cls_orig']:
    rep(f'drop_s2+s3 +{k}',np.column_stack([Bd,NEW[k][0]]),np.column_stack([BdT,NEW[k][1]]))
# ---- gated mix for hiexp ----
QS=[0.1,0.2,0.3,0.4]; WS=np.round(np.arange(0,1.01,0.1),2)
hx,hxT=NEW['d10_hiexp']
def gate(p,h,thr,w): g=p>=thr; q=p.copy(); q[g]=(1-w)*p[g]+w*h[g]; return q
def fit_gate(idx):
    # inner 5-fold on idx: base NNLS fit on inner-train, predict inner-val; choose (q,w) minimizing inner RMSE
    pin=np.zeros(len(idx)); thr_in=np.zeros(len(idx)); S={}
    for itr,iva in KFold(5,shuffle=True,random_state=7).split(idx):
        wt=nnls(A1(B[idx[itr]]),y[idx[itr]])[0]; ptr=A1(B[idx[itr]])@wt; pva=A1(B[idx[iva]])@wt
        for q in QS:
            thr=np.quantile(ptr,1-q)
            for w in WS: S.setdefault((q,w),[]).append(((gate(pva,hx[idx[iva]],thr,w)-y[idx[iva]])**2).sum())
    return min(S,key=lambda k: sum(S[k]))
gres=[]; picks=[]
for s in range(10):
    pp=np.zeros(len(y)); pb=np.zeros(len(y))
    for tr,va in KFold(5,shuffle=True,random_state=100+s).split(B):
        q,w=fit_gate(tr); picks.append((q,w))
        wt=nnls(A1(B[tr]),y[tr])[0]; ptr=A1(B[tr])@wt; pva=A1(B[va])@wt
        pb[va]=pva; pp[va]=gate(pva,hx[va],np.quantile(ptr,1-q),w)
    gres.append((r(pp,y),(pp-y)[hi].mean(),(pp-y)[lo].mean(),r(pb,y)))
g=np.array(gres).mean(0)
from collections import Counter
pc=Counter(picks).most_common(5)
s=f'| d10_hiexp GATED (q,w by inner CV) | {g[0]:.3f} | {g[0]-base[0]:+.3f} | {g[1]:.1f} | {g[2]:.1f} | - | picks top5={pc} |'
print(s); L.append(s)
# fixed-grid diagnostic (non-nested, optimistic) for transparency
L+=['','#### gated diag (non-nested OOF of base full-fit, optimistic): RMSE by (q,w)','| q | best w | RMSE | hi |','|---|---|---|---|']
pfull=A1(B)@fw(B)
for q in QS:
    thr=np.quantile(pfull,1-q); best=min(WS,key=lambda w:r(gate(pfull,hx,thr,w),y)); pg=gate(pfull,hx,thr,best)
    L.append(f'| {q} | {best} | {r(pg,y):.3f} | {(pg-y)[hi].mean():.1f} |')
L.append(f'(base full-fit in-sample RMSE {r(pfull,y):.3f})')
best=min(rows,key=lambda z:z[1]); bstr=f'BEST: {best[0]} nCV={best[1]:.3f} (Δ {best[1]-base[0]:+.3f}); gated={g[0]:.3f}'
print(bstr); L.append(''); L.append(bstr)
if best[1]<=12.55:
    X,XT=best[2],best[3]; w=fw(X); pt=A1(XT)@w; s2=sub.copy(); s2.iloc[:,1]=pt
    fn=f'{H}/cand_{best[0].replace("+","p").replace(" ","_")}_cv{best[1]:.3f}.csv'; s2.to_csv(fn,index=False); L.append(f'CANDIDATE {fn}'); print('CAND',fn)
else: L.append('no candidate (best nCV > 12.55)'); print('no candidate')
if g[0]<=base[0]-0.05: L.append('GATED improvement ≥0.05 → consider candidate (not auto-written)')
open(f'{H}/ev_d8d10.md','w').write('\n'.join(L)+'\n')
