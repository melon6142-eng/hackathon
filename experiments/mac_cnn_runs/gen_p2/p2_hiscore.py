# (3) 기존 vs 새 고경도 점수(OOF 회귀값) on 실제 18 고경도: AUPRC, P@18, 이미지별 순위 변화. fold seed 별
import numpy as np,csv,os,sys
from sklearn.metrics import average_precision_score as ap
y=np.array([float(r['hardness']) for r in csv.DictReader(open('~/AI/daker/open/train.csv'))]);hi=y>230;H=np.where(hi)[0]
out=['## (3) 고경도 점수 비교 (점수 = 3seed 평균 OOF 예측값)','| fold rs | 점수 | AUPRC | P@18 | 18장 안 ρ |','|---|---|---|---|---|'];ranks={}
from scipy.stats import spearmanr
for rs in ('42','7','123','2026'):
    B=['../dinoB_orig','../dinoB_orig_s1','../dinoB_orig_s2'] if rs=='42' else [f'../indepfold/rs{rs}_orig_s{s}' for s in (42,1,2)]
    N=[f'ft/pre_rs{rs}_s{s}' for s in (42,1,2)]
    if not all(os.path.exists(d+'/oof.npy') for d in N): continue
    for nm,D in (('기존 ViT orig',B),('새(param-pretrain)',N)):
        s=np.mean([np.load(d+'/oof.npy') for d in D],0);rk=(-s).argsort().argsort()+1;ranks[(rs,nm)]=rk[H]
        out.append(f'| {rs} | {nm} | {ap(hi,s):.3f} | {hi[np.argsort(-s)[:18]].mean():.3f} | {spearmanr(s[H],y[H])[0]:.2f} |')
out.append('\n### 고경도 18장 순위(전체 500 중, 작을수록 높게 예측) 기존→새\n| idx | 경도 | '+' | '.join(r for r in ('42','7','123','2026') if (r,'기존 ViT orig') in ranks)+' |')
out.append('|'+'---|'*(2+sum((r,'기존 ViT orig') in ranks for r in ('42','7','123','2026'))))
for j,i in enumerate(H[np.argsort(-y[H])]):
    jj=list(H).index(i);out.append(f'| {i} | {y[i]:.0f} | '+' | '.join(f"{ranks[(r,'기존 ViT orig')][jj]}→{ranks[(r,'새(param-pretrain)')][jj]}" for r in ('42','7','123','2026') if (r,'기존 ViT orig') in ranks)+' |')
print('\n'.join(out));open('p2_hiscore.md','w').write('\n'.join(out)+'\n')
