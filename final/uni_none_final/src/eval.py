# 4 fold seeds x 3 train seeds: uni_none-blend (BL1 + uni_none3 + trans3) vs final_clean (BL1 + orig3 + trans3). Writes eval.md/json + decision.md
import os,json,numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score as ap
from common import *
Y=y(); FC='~/AI/daker/final_clean'; BD='~/AI/daker/cnn/runs/bal_d4'
def ncv(cols,reps=20):
    X=np.column_stack(cols+[np.ones(500)]);o=[]
    for r in range(reps):
        p=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+r).split(X): w,_=nnls(X[tr],Y[tr]);p[va]=X[va]@w
        o.append(rm(p,Y))
    return float(np.mean(o))
def met(p): return dict(rmse=rm(p,Y),hi=hb(p,Y),lo=lb(p,Y),auprc=float(ap(Y>230,p)),p18=float((Y[np.argsort(-p)[:18]]>230).mean()))
def fcv(ch,s,fs): return np.load(f'{FC}/vit/{ch}_s{s}'+('' if fs==42 else f'_fs{fs}')+'/oof.npy')
def uni(s,fs,src):
    if fs==42: return np.load(f'{BD}/arms/uni_none_s{s}/oof.npy' if src=='bal_d4' else f'{O}/arms/ckpt/uni_none_s{s}/oof.npy')
    if s==42: return np.load(f'{BD}/arms_fs/uni_none_s42_fs{fs}/oof.npy')
    return np.load(f'{O}/arms/val/uni_none_s{s}_fs{fs}/oof.npy')
R={};L=['# uni_none 블렌드 vs final_clean — fold seed 4개 × 학습 seed 3개','','eq3 = (BL1 + ViT_A 3seed평균 + ViT trans 3seed평균)/3, ViT_A = orig(final_clean) 또는 uni_none. NNLS nCV = 20회 반복 5-fold NNLS(+절편).','',
 '| fold seed | 모델 | uni단독3seed | eq3 | Δeq3 | NNLS nCV | ΔnCV | hi bias | lo bias | AUPRC | P@18 |','|---|---|---|---|---|---|---|---|---|---|---|']
for fs in [42,7,123,2026]:
    b=np.load(f'{FC}/bl1/oof_fs{fs}.npy'); t=np.mean([fcv('trans',s,fs) for s in (42,1,2)],0); o=np.mean([fcv('orig',s,fs) for s in (42,1,2)],0)
    u=np.mean([uni(s,fs,'bal_d4') for s in (42,1,2)],0)
    e_fc,e_u=(b+o+t)/3,(b+u+t)/3; r=dict(fc=dict(eq3=met(e_fc),ncv=ncv([b,o,t]),member=rm(o,Y)),uni=dict(eq3=met(e_u),ncv=ncv([b,u,t]),member=rm(u,Y),
        member_ind=[rm(uni(s,fs,'bal_d4'),Y) for s in (42,1,2)]))
    if fs==42 and all(os.path.exists(f'{O}/arms/ckpt/uni_none_s{s}/oof.npy') for s in (42,1,2)):
        uc=np.mean([uni(s,42,'ckpt') for s in (42,1,2)],0); r['uni_ckpt']=dict(eq3=met((b+uc+t)/3),ncv=ncv([b,uc,t]),member=rm(uc,Y))
    r['d_eq3']=r['uni']['eq3']['rmse']-r['fc']['eq3']['rmse']; r['d_ncv']=r['uni']['ncv']-r['fc']['ncv']; R[fs]=r
    for k,n in [('fc','final_clean (orig)'),('uni','uni_none')]+([('uni_ckpt','uni_none 재학습(ckpt)')] if 'uni_ckpt' in r else []):
        m=r[k]['eq3']; d1=f"{m['rmse']-r['fc']['eq3']['rmse']:+.3f}" if k!='fc' else '–'; d2=f"{r[k]['ncv']-r['fc']['ncv']:+.3f}" if k!='fc' else '–'
        L.append(f"| {fs} | {n} | {r[k]['member']:.3f} | {m['rmse']:.3f} | {d1} | {r[k]['ncv']:.3f} | {d2} | {m['hi']:+.1f} | {m['lo']:+.1f} | {m['auprc']:.3f} | {m['p18']:.3f} |")
fss=[42,7,123,2026]; wins=sum(R[f]['d_eq3']<0 for f in fss); md=float(np.mean([R[f]['d_eq3'] for f in fss])); mn=float(np.mean([R[f]['d_ncv'] for f in fss]))
mean=lambda k,q: float(np.mean([R[f][k]['eq3'][q] for f in fss]))
L+=['',f"- 4 seed 평균 eq3: final_clean {mean('fc','rmse'):.3f} / uni_none {mean('uni','rmse'):.3f} (Δ {md:+.3f}), uni 승 {wins}/4",
    f"- 4 seed 평균 NNLS nCV: final_clean {np.mean([R[f]['fc']['ncv'] for f in fss]):.3f} / uni_none {np.mean([R[f]['uni']['ncv'] for f in fss]):.3f} (Δ {mn:+.3f})",
    f"- 4 seed 평균 hi/lo bias: fc {mean('fc','hi'):+.1f}/{mean('fc','lo'):+.1f}, uni {mean('uni','hi'):+.1f}/{mean('uni','lo'):+.1f}; AUPRC fc {mean('fc','auprc'):.3f} / uni {mean('uni','auprc'):.3f}; P@18 fc {mean('fc','p18'):.3f} / uni {mean('uni','p18'):.3f}"]
ok=wins>=3 and md<=-0.03
json.dump(dict(R={str(k):v for k,v in R.items()},wins=wins,mean_d_eq3=md,mean_d_ncv=mn,uni_main=ok),open(f'{O}/eval.json','w'),indent=1)
open(f'{O}/eval.md','w').write('\n'.join(L))
ck={}
for s in (42,1,2):
    p=f'{O}/arms/ckpt/uni_none_s{s}/ckpt_check.json'
    if os.path.exists(p): ck[s]=json.load(open(p))
D=['# decision — uni_none vs final_clean','',f'규칙(사전 고정): 3seed eq3에서 uni_none이 fold seed 4개 중 3개 이상에서 final_clean보다 낮고, 평균 Δ ≤ −0.03이면 uni_none 메인, 아니면 final_clean 메인.','',
   f'- 결과: uni 승 {wins}/4, 평균 Δeq3 {md:+.3f}, 평균 ΔnCV {mn:+.3f}',
   f"- 판정: **{'uni_none 메인' if ok else 'final_clean 메인'}**",
   f"- 메인 CSV 후보: {'submission_uninone_eq3_ckpt.csv (ckpt 재추론, 없으면 submission_uninone_eq3.csv)' if ok else '~/AI/daker/final_clean/submission_final_clean_eq3.csv (또는 NNLS판 submission_final_clean.csv)'}",'','## ckpt 재학습 확인 (fold seed 42)']
for s,c in ck.items(): D.append(f"- seed {s}: OOF bit-identical={c['vs_bal_d4']['bit_identical_oof']}, OOF {c['vs_bal_d4']['oof_rmse_new']:.3f} (기존 {c['vs_bal_d4']['oof_rmse_bal_d4']:.3f}), test corr vs 기존 {c['vs_bal_d4']['test_corr']:.6f}, max|Δ| {c['vs_bal_d4']['test_maxabs']:.4f}; ckpt 재추론 vs 학습중 예측 max|Δ| {c['test_ckpt_vs_train_time_maxabs']:.2e}")
D+=['','제출하지 않음. 상세 표는 eval.md.']
open(f'{O}/decision.md','w').write('\n'.join(D)); print('\n'.join(L+['']+D))
