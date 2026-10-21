# Final blend = NNLS(with intercept column) on OOFs of [BL1, mean(orig 3 seeds), mean(trans 3 seeds)]  (same method as runs/dino_ft2/eval3.py)
import numpy as np,pandas as pd,json,os,glob
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from common import *
Y=y(); OLD='~/AI/daker/cnn/runs/repro_vit/ref'; OLDV='~/AI/daker/cnn/runs'
def ncv(cols,reps=20):
    X=np.column_stack(cols+[np.ones(500)]);o=[]
    for s in range(reps):
        p=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X): w,_=nnls(X[tr],Y[tr]);p[va]=X[va]@w
        o.append(rm(p,Y))
    return float(np.mean(o)),float(np.std(o))
def vit(chan,fs):
    tags=[f'{chan}_s{s}' + ('' if fs==42 else f'_fs{fs}') for s in (42,1,2)]
    ds=[f'{O}/vit/{t}' for t in tags]
    if not all(os.path.exists(f'{d}/oof.npy') for d in ds): return None
    return np.mean([np.load(f'{d}/oof.npy') for d in ds],0),np.mean([np.load(f'{d}/test.npy') for d in ds],0),[rm(np.load(f'{d}/oof.npy'),Y) for d in ds]
L=['# final_clean 결과','']; R_={}
# ---- main: fold seed 42
b=np.load(f'{O}/bl1/oof_fs42.npy'); bt=np.load(f'{O}/bl1/test.npy'); vo=vit('orig',42); vt=vit('trans',42)
X=np.column_stack([b,vo[0],vt[0],np.ones(500)]); w=nnls(X,Y)[0]; oof=X@w
pt=np.column_stack([bt,vo[1],vt[1],np.ones(1000)])@w; m,s=ncv([b,vo[0],vt[0]])
eq=(b+vo[0]+vt[0])/3
# old references
ob=np.load(f'{OLD}/oof_blend1.npy'); old=pd.read_csv(f'{OLD}/cand_BL1_Bavgorig-trans_cv12.61.csv')
oo=np.mean([np.load(f'{OLDV}/{t}/oof.npy') for t in ['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2']],0)
ot=np.mean([np.load(f'{OLDV}/{t}/oof.npy') for t in ['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2']],0)
Xo=np.column_stack([ob,oo,ot,np.ones(500)]); wo=nnls(Xo,Y)[0]; mo,so=ncv([ob,oo,ot])
row=lambda n,p,q: f'| {n} | {rm(p,Y):.3f} | {hb(p,Y):+.1f} | {lb(p,Y):+.1f} | {rm(q,Y):.3f} | {hb(q,Y):+.1f} | {lb(q,Y):+.1f} |'
L+=['## fold seed 42 (KFold(5,shuffle,rs=42), 원래와 같은 분할)','','| 멤버 | 새 OOF | 새 hi편향 | 새 lo편향 | 옛 OOF | 옛 hi | 옛 lo |','|---|---|---|---|---|---|---|',
    row('BL1',b,ob),row('ViT orig 3seed평균',vo[0],oo),row('ViT trans 3seed평균',vt[0],ot),
    row('블렌드 NNLS(in-sample)',oof,Xo@wo),row('블렌드 균등평균 eq3',eq,(ob+oo+ot)/3),'',
    f'- ViT 개별 OOF orig {np.round(vo[2],3).tolist()} / trans {np.round(vt[2],3).tolist()}',
    f'- 블렌드 nCV(20회 NNLS) 새 {m:.3f}±{s:.3f} / 옛 {mo:.3f}±{so:.3f}',
    f'- 가중치 새 BL1/orig/trans/c = {np.round(w,4).tolist()} / 옛 {np.round(wo,4).tolist()}',
    '- 주의: 옛 BL1 OOF는 KFold rs 42/43/44 3회 평균 + 전체 train 라벨로 고른 poly top-14(OOF 낙관 편향), 새 것은 rs=42 1회 + fold 내부 선택.']
c=np.corrcoef(pt,old.hardness.values)[0,1]
L+=[f'- test: 옛 11.33 후보와 corr {c:.5f}, 평균 차 {pt.mean()-old.hardness.mean():+.2f}, max|diff| {np.abs(pt-old.hardness.values).max():.2f}, RMS diff {np.sqrt(((pt-old.hardness.values)**2).mean()):.2f}',
    f'- test 평균/표준편차 새 {pt.mean():.2f}/{pt.std():.2f}, 옛 {old.hardness.mean():.2f}/{old.hardness.std():.2f}']
ss=pd.read_csv(f'{R}/sample_submission.csv'); sub=pd.DataFrame({'ID':test_ids(),'hardness':pt})
assert list(sub.ID)==list(ss.ID) and list(sub.columns)==list(ss.columns) and sub.hardness.notna().all() and len(sub)==1000
fn=f'{O}/submission_final_clean.csv'; sub.to_csv(fn,index=False); L+=[f'- 최종 CSV {fn} sha256 {sha(fn)}']
pd.DataFrame({'ID':test_ids(),'hardness':(bt+vo[1]+vt[1])/3}).to_csv(f'{O}/submission_final_clean_eq3.csv',index=False)
R_['fs42']=dict(nnls_insample=rm(oof,Y),ncv=m,eq3=rm(eq,Y),w=w.tolist(),test_corr_old=float(c))
# ---- other fold seeds
L+=['','## fold seed별 (멤버 OOF를 해당 분할로 새로 학습)','','| fold seed | BL1 | orig | trans | eq3 균등평균 | NNLS nCV |','|---|---|---|---|---|---|']
E=[];N=[]
for fs in [42,7,123,2026]:
    p=f'{O}/bl1/oof_fs{fs}.npy'; a_=vit('orig',fs); t_=vit('trans',fs)
    if not os.path.exists(p) or a_ is None or t_ is None: L.append(f'| {fs} | (미완) | | | | |'); continue
    bb=np.load(p); e=rm((bb+a_[0]+t_[0])/3,Y); n=ncv([bb,a_[0],t_[0]])[0]; E.append(e); N.append(n)
    L.append(f'| {fs} | {rm(bb,Y):.3f} | {rm(a_[0],Y):.3f} | {rm(t_[0],Y):.3f} | {e:.3f} | {n:.3f} |')
if E: L+=[f'- 완료된 {len(E)}개 seed 평균: eq3 {np.mean(E):.3f}, NNLS nCV {np.mean(N):.3f} (옛 eq3 4-seed 평균 12.667; 옛 split별 12.539/12.795/12.538/12.796 — 옛 값은 BL1 라벨 선택 낙관 편향 포함)']
R_['seeds']={'eq3':E,'ncv':N}
json.dump(R_,open(f'{O}/results.json','w'),indent=1); open(f'{O}/results.md','w').write('\n'.join(L)); print('\n'.join(L))
