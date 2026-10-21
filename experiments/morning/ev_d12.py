import numpy as np, pandas as pd, os
from scipy.optimize import nnls
from sklearn.model_selection import KFold
D='./runs'; H=f'{D}/overnight'; M=f'{D}/morning'
y=pd.read_csv('./open/train.csv')['hardness'].values
sub=pd.read_csv('./out/submission_blend1_cv12.95.csv')
ld=lambda t,k: np.load(f'{D}/{t}/{k}.npy'); av=lambda ts,k: np.mean([ld(t,k) for t in ts],0)
r=lambda p,t: np.sqrt(((p-t)**2).mean()); hi=y>230; lo=y<170
O3=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','dino_ft2/dinoB_orig_s2']; T3=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','dino_ft2/dinoB_trans_s2']
B=np.column_stack([np.load(f'{D}/oof_blend1.npy'),av(O3,'oof'),av(T3,'oof')]); BT=np.column_stack([sub.iloc[:,1].values,av(O3,'test'),av(T3,'test')])
P={'ml42':'overnight/d11_multilayer','dldl':'overnight/d8_dldl_orig','canon':'overnight/d10_canon'}
for k in ['d12_ml_orig_s1','d12_ml_orig_s2','d12_ml_trans_s42','d12_ml_asym','d12_ml_orig_s42_frs7']:
    if os.path.exists(f'{M}/{k}/oof.npy'): P[k]=f'morning/{k}'
N={k:(ld(v,'oof'),ld(v,'test')) for k,v in P.items()}
A1=lambda X: np.column_stack([X,np.ones(len(X))])
def ncv(X,R=10):
    o=[]
    for s in range(R):
        pp=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X): pp[va]=A1(X[va])@nnls(A1(X[tr]),y[tr])[0]
        o.append((r(pp,y),(pp-y)[hi].mean(),(pp-y)[lo].mean()))
    o=np.array(o); return o[:,0].mean(),o[:,0].std(),o[:,1].mean(),o[:,2].mean()
fw=lambda X: nnls(A1(X),y)[0]
L=['### Solo OOF','| tag | OOF | hi | lo | corr Bo / ml42 |','|---|---|---|---|---|']
for k in N:
    if k.startswith('d12'):
        o=N[k][0]; L.append(f'| {k} | {r(o,y):.3f} | {(o-y)[hi].mean():.1f} | {(o-y)[lo].mean():.1f} | {np.corrcoef(B[:,1],o)[0,1]:.3f} / {np.corrcoef(N["ml42"][0],o)[0,1]:.3f} |')
mlo=[k for k in ['ml42','d12_ml_orig_s1','d12_ml_orig_s2'] if k in N]
N['mlavg']=(np.mean([N[k][0] for k in mlo],0),np.mean([N[k][1] for k in mlo],0))
if 'd12_ml_trans_s42' in N: N['mlavg+tr']=tuple((len(mlo)*N['mlavg'][i]+N['d12_ml_trans_s42'][i])/(len(mlo)+1) for i in (0,1))
o=N['mlavg'][0]; L.append(f'| mlavg({len(mlo)}) | {r(o,y):.3f} | {(o-y)[hi].mean():.1f} | {(o-y)[lo].mean():.1f} | {np.corrcoef(B[:,1],o)[0,1]:.3f} / - |')
def oa(ks,tks=()):  # add orig-type into Borig avg, trans-type into Btrans avg
    X=B.copy();XT=BT.copy()
    for j,kk in ((1,ks),(2,tks)):
        n=3+len(kk); X[:,j]=(3*B[:,j]+sum(N[k][0] for k in kk))/n; XT[:,j]=(3*BT[:,j]+sum(N[k][1] for k in kk))/n
    return X,XT
base=ncv(B); rows=[]
L+=['','### nCV 10×5 (NNLS+icpt), Δ vs 12.601 base','| 구성 | nCV | sd | Δ | hi | lo |','|---|---|---|---|---|---|']
def rep(lab,X,XT):
    m=ncv(X); L.append(f'| {lab} | {m[0]:.3f} | {m[1]:.3f} | {m[0]-base[0]:+.3f} | {m[2]:.1f} | {m[3]:.1f} |'); rows.append((lab,m[0],X,XT)); return m[0]
rep('base',B,BT)
NB=['ml42','dldl','canon']; nb=rep('night best: Bo←ml42+dldl+canon',*oa(NB))
extra=[k for k in N if k.startswith('d12') and 'frs7' not in k]
for k in extra:
    if 'trans' in k: rep(f'night best +{k}(→Btrans)',*oa(NB,[k]))
    else: rep(f'night best +{k}(→Borig)',*oa(NB+[k]))
    rep(f'night best +{k}(new col)',np.column_stack([oa(NB)[0],N[k][0]]),np.column_stack([oa(NB)[1],N[k][1]]))
rep('base +mlavg(col)',np.column_stack([B,N['mlavg'][0]]),np.column_stack([BT,N['mlavg'][1]]))
rep('Bo←mlseeds',*oa(mlo))
rep('Bo←mlseeds+dldl+canon',*oa(mlo+['dldl','canon']))
if 'd12_ml_trans_s42' in N: rep('Bo←mlseeds+dldl+canon, Bt←mltrans',*oa(mlo+['dldl','canon'],['d12_ml_trans_s42']))
if 'd12_ml_asym' in N:
    rep('Bo←mlseeds+dldl+canon +asym(col)',np.column_stack([oa(mlo+['dldl','canon'])[0],N['d12_ml_asym'][0]]),np.column_stack([oa(mlo+['dldl','canon'])[1],N['d12_ml_asym'][1]]))
    if 'd12_ml_trans_s42' in N: rep('ALL: Bo←mlseeds+dldl+canon, Bt←mltrans, +asym(col)',np.column_stack([oa(mlo+['dldl','canon'],['d12_ml_trans_s42'])[0],N['d12_ml_asym'][0]]),np.column_stack([oa(mlo+['dldl','canon'],['d12_ml_trans_s42'])[1],N['d12_ml_asym'][1]]))
# all-seed (no post-hoc drop): Borig/Btrans = s0,s1,s2,s3 avg
O4=O3+['overnight/dinoB_orig_s3'];T4=T3+['overnight/dinoB_trans_s3']
B4=np.column_stack([B[:,0],av(O4,'oof'),av(T4,'oof')]);B4T=np.column_stack([BT[:,0],av(O4,'test'),av(T4,'test')])
def oa4(ks,tks=()):
    X=B4.copy();XT=B4T.copy()
    for j,kk in ((1,ks),(2,tks)):
        n=4+len(kk);X[:,j]=(4*B4[:,j]+sum(N[k][0] for k in kk))/n;XT[:,j]=(4*B4T[:,j]+sum(N[k][1] for k in kk))/n
    return X,XT
L.append('| -- 전 시드(s0~s3) 평균, 사후 drop 없음 -- |||||| ')
rep('[all-seed] base',B4,B4T)
rep('[all-seed] Bo←ml42+dldl+canon',*oa4(['ml42','dldl','canon']))
rep('[all-seed] Bo←mlseeds+dldl+canon',*oa4(mlo+['dldl','canon']))
if 'd12_ml_trans_s42' in N: rep('[all-seed] Bo←mlseeds+dldl+canon, Bt←mltrans',*oa4(mlo+['dldl','canon'],['d12_ml_trans_s42']))
if 'd12_ml_orig_s42_frs7' in N:
    o7=N['d12_ml_orig_s42_frs7'][0];o42=N['ml42'][0]
    L.append(f'| -- fold rs 비교: ml s42 solo OOF rs42={r(o42,y):.3f} / rs7={r(o7,y):.3f} (hi {(o42-y)[hi].mean():.1f}/{(o7-y)[hi].mean():.1f}), corr {np.corrcoef(o42,o7)[0,1]:.3f} -- ||||||')
    N['ml7']=N['d12_ml_orig_s42_frs7']
    rep('night best, ml42→ml(rs7)',*oa(['ml7','dldl','canon']))
    rep('base +ml(rs7) col',np.column_stack([B,o7]),np.column_stack([BT,N['ml7'][1]]))
# all-seed (no post-hoc drop): Borig/Btrans = s0,s1,s2,s3 avg
O4=O3+['overnight/dinoB_orig_s3'];T4=T3+['overnight/dinoB_trans_s3']
B4=np.column_stack([B[:,0],av(O4,'oof'),av(T4,'oof')]);B4T=np.column_stack([BT[:,0],av(O4,'test'),av(T4,'test')])
def oa4(ks,tks=()):
    X=B4.copy();XT=B4T.copy()
    for j,kk in ((1,ks),(2,tks)):
        n=4+len(kk);X[:,j]=(4*B4[:,j]+sum(N[k][0] for k in kk))/n;XT[:,j]=(4*B4T[:,j]+sum(N[k][1] for k in kk))/n
    return X,XT
L.append('| -- 전 시드(s0~s3) 평균, 사후 drop 없음 -- |||||| ')
rep('[all-seed] base',B4,B4T)
rep('[all-seed] Bo←ml42+dldl+canon',*oa4(['ml42','dldl','canon']))
rep('[all-seed] Bo←mlseeds+dldl+canon',*oa4(mlo+['dldl','canon']))
if 'd12_ml_trans_s42' in N: rep('[all-seed] Bo←mlseeds+dldl+canon, Bt←mltrans',*oa4(mlo+['dldl','canon'],['d12_ml_trans_s42']))
if 'd12_ml_orig_s42_frs7' in N:
    o7=N['d12_ml_orig_s42_frs7'][0];o42=N['ml42'][0]
    L.append(f'| -- fold rs 비교: ml s42 solo OOF rs42={r(o42,y):.3f} / rs7={r(o7,y):.3f} (hi {(o42-y)[hi].mean():.1f}/{(o7-y)[hi].mean():.1f}), corr {np.corrcoef(o42,o7)[0,1]:.3f} -- ||||||')
    N['ml7']=N['d12_ml_orig_s42_frs7']
    rep('night best, ml42→ml(rs7)',*oa(['ml7','dldl','canon']))
    rep('base +ml(rs7) col',np.column_stack([B,o7]),np.column_stack([BT,N['ml7'][1]]))
# robustness: replace ml42 in 12.563 config by each single model; repeat-wise and sample-wise win vs base
def preds(X,R=10):
    P=[]
    for s_ in range(R):
        pp=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+s_).split(X): pp[va]=A1(X[va])@nnls(A1(X[tr]),y[tr])[0]
        P.append(pp)
    return np.array(P)
PB=preds(B); eb=((PB-y)**2)
L+=['','### 재현성: 12.563 구성의 ml42 자리를 단일 모델로 대체 (vs base)','| 대체 | nCV | Δ | repeat win(10) | 샘플 SSE win율 | hi |','|---|---|---|---|---|---|']
cands=[('ml42',[ 'ml42'],[])]+[(k,[k],[]) for k in ['d12_ml_orig_s1','d12_ml_orig_s2','ml7'] if k in N]
if 'd12_ml_trans_s42' in N: cands.append(('trans_s42→Bt',[],['d12_ml_trans_s42']))
for lab,ko,kt in cands:
    X,_=oa(ko+['dldl','canon'],kt); PC=preds(X); ec=((PC-y)**2)
    rr=np.sqrt(ec.mean(1)); rb=np.sqrt(eb.mean(1))
    L.append(f'| {lab} | {rr.mean():.3f} | {rr.mean()-rb.mean():+.3f} | {(rr<rb).sum()}/10 | {(ec.mean(0)<eb.mean(0)).mean():.3f} | {(PC.mean(0)-y)[hi].mean():.1f} |')
X,_=oa(['dldl','canon']); PC=preds(X); rr=np.sqrt(((PC-y)**2).mean(1))
L.append(f'| (ml 없음: dldl+canon만) | {rr.mean():.3f} | {rr.mean()-np.sqrt(eb.mean(1)).mean():+.3f} | {(rr<np.sqrt(eb.mean(1))).sum()}/10 | {(((PC-y)**2).mean(0)<eb.mean(0)).mean():.3f} | {(PC.mean(0)-y)[hi].mean():.1f} |')
b=min(rows,key=lambda z:z[1]); L+=['',f'BEST: {b[0]} nCV={b[1]:.3f}']
if b[1]<=12.55:
    s2=sub.copy(); s2.iloc[:,1]=A1(b[3])@fw(b[2]); fn=f'{M}/cand_d12_cv{b[1]:.3f}.csv'; s2.to_csv(fn,index=False); L.append(f'CANDIDATE {fn} w={np.round(fw(b[2]),3).tolist()}')
else: L.append('no candidate (>12.55)')
open(f'{M}/ev_d12.md','w').write('\n'.join(L)+'\n'); print('\n'.join(L))
