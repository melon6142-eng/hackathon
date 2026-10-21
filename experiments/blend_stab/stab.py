import numpy as np,pandas as pd,glob,json
from scipy.optimize import nnls
from sklearn.model_selection import KFold
D='./runs';S='src/runs/indepfold'
y=pd.read_csv('./open/train.csv').hardness.values;hi=y>230
r=lambda p:np.sqrt(((p-y)**2).mean())
av=lambda ts,k:np.mean([np.load(f'{D}/{t}/{k}.npy') for t in ts],0)
O3=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','dino_ft2/dinoB_orig_s2'];T3=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','dino_ft2/dinoB_trans_s2']
X={'base42':np.column_stack([np.load(f'{D}/oof_blend1.npy'),av(O3,'oof'),av(T3,'oof')])}
for rs in (7,123,2026): X[f'rs{rs}']=np.load(f'{S}/blendX_rs{rs}.npy')
K=list(X);A1=lambda M:np.column_stack([M,np.ones(len(M))])
fit=lambda M,t:nnls(A1(M),t)[0]
EQ=np.array([1/3,1/3,1/3,0.])
def eqfit(M,t):  # equal weights + fitted intercept
    return np.r_[EQ[:3],(t-M@EQ[:3]).mean()]
def shr(lam):
    def f(M,t):
        w=fit(M,t);ws=lam*w[:3]+(1-lam)*EQ[:3];return np.r_[ws,(t-M@ws).mean()]
    return f
M_={'perseed_NNLS':fit,'equal+b':eqfit,'shrink0.5':shr(.5),'shrink0.25':shr(.25)}
def ncv(h,fitter,others,R=10):
    # strict: weights fit only on train idx (of nCV split of held-out seed h), using OOFs of `others` seeds (pooled rows); scored on h's va idx
    out=[];Xh=X[h]
    for s in range(R):
        pp=np.zeros(len(y))
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(y):
            M=np.vstack([X[k][tr] for k in others]);t=np.tile(y[tr],len(others))
            pp[va]=A1(Xh[va])@fitter(M,t)
        out.append(r(pp))
    return np.mean(out)
res={};L=['# 블렌드 안정화 (multi-fold-seed NNLS) — 결과','',
 '데이터: fold seed 4종의 블렌드 입력 [BL1, ViT-B orig 3시드 평균, trans 3시드 평균] (base42 = 12.601 기준 split, rs7/123/2026 = Exp D 독립 fold). train 500만 사용, test 미사용.','',
 '평가: 각 held-out fold seed h의 OOF에 대해 nCV(5-fold×10, rs 100+s). 가중치는 **train idx만** 사용해 적합, va idx에서 채점.',
 '- self: 가중치를 h 자신의 OOF(train idx)로 적합 = 현행 방식 (12.601 계산법)',
 '- cross: 가중치를 h를 제외한 다른 fold seed들의 OOF(train idx 행 pooled)로 적합 → h에서 채점 (fold-seed 일반화)',
 '- all: h 포함 4개 seed pooled','']
L+=['| 방식 | 적합 소스 | '+' | '.join(K)+' | 독립3 평균±sd | 4 평균 |','|---|---|'+'---|'*len(K)+'---|---|']
for mn,f in M_.items():
  for src in ('self','cross','all'):
    v=[]
    for h in K:
      oth=[h] if src=='self' else [k for k in K if k!=h] if src=='cross' else K
      if mn=='equal+b' and src!='self': oth=[h] if False else oth
      v.append(ncv(h,f,oth))
    i3=np.array(v[1:]);res[f'{mn}|{src}']=v
    L.append(f'| {mn} | {src} | '+' | '.join(f'{a:.3f}' for a in v)+f' | {i3.mean():.3f}±{i3.std(ddof=1):.3f} | {np.mean(v):.3f} |')
    if mn=='equal+b': break   # equal weights don't depend on source beyond intercept
# simple mean (no intercept) too
v=[r(X[h]@EQ[:3]) for h in K];L.append(f'| 단순평균(절편X) | - | '+' | '.join(f'{a:.3f}' for a in v)+f' | {np.mean(v[1:]):.3f}±{np.std(v[1:],ddof=1):.3f} | {np.mean(v):.3f} |')
L+=['','## 가중치 안정성 (각 seed 전체 OOF로 NNLS+절편)','| seed | BL1 | orig | trans | 절편 | 합 |','|---|---|---|---|---|---|']
W=[]
for h in K:
  w=fit(X[h],y);W.append(w);L.append(f'| {h} | '+' | '.join(f'{a:.3f}' for a in w[:3])+f' | {w[3]:.2f} | {w[:3].sum():.3f} |')
W=np.array(W);L.append(f'| sd | '+' | '.join(f'{a:.3f}' for a in W[:,:3].std(0,ddof=1))+' | | |')
Mall=np.vstack([X[k] for k in K]);wp=fit(Mall,np.tile(y,4));L.append(f'| pooled4 | '+' | '.join(f'{a:.3f}' for a in wp[:3])+f' | {wp[3]:.2f} | {wp[:3].sum():.3f} |')
# bootstrap weight stability on base42
rng=np.random.default_rng(0);B=[]
for _ in range(300):
  i=rng.integers(0,500,500);B.append(fit(X['base42'][i],y[i])[:3])
L.append(f'| base42 bootstrap sd(300) | '+' | '.join(f'{a:.3f}' for a in np.std(B,0))+' | | |')
# fixed-weight transfer: full-fit weight from seed a scored on seed b
L+=['','## 고정 가중치 전이 (행: 가중치 적합 seed 전체 OOF, 열: 채점 seed, 낙관적—같은 y)','| fit\\eval | '+' | '.join(K)+' |','|---|'+'---|'*len(K)]
for a,w in zip(K,W): L.append(f'| {a} | '+' | '.join(f'{r(A1(X[b])@w):.3f}' for b in K)+' |')
L.append(f'| pooled4 | '+' | '.join(f'{r(A1(X[b])@wp):.3f}' for b in K)+' |')
open('results.md','w').write('\n'.join(L)+'\n');json.dump({'res':res,'W':W.tolist(),'wp':wp.tolist()},open('results.json','w'))
print('\n'.join(L))
