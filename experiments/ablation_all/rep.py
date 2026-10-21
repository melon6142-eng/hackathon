import pickle,numpy as np,pandas as pd
y=pd.read_csv('./open/train.csv').hardness.values;hi=y>230;lo=y<170
R=pickle.load(open('raw.pkl','rb'));K=list(R[0])
rm=lambda p:np.sqrt(((p-y)**2).mean())
S={k:np.array([rm(r[k]) for r in R]) for k in K}
bi={k:np.mean([(r[k]-y)[hi].mean() for r in R]) for k in K};bl={k:np.mean([(r[k]-y)[lo].mean() for r in R]) for k in K}
best=np.mean([r['B|최고블렌드'] for r in R],0);res=y-best
basep=np.mean([r['A|base'] for r in R],0);resb=y-basep
def row(k,ref):
    d=S[k]-S[ref];sd=S[ref].std();sig='예' if abs(d.mean())>sd else '아니오'
    return f'| {k.split("|",1)[1]} | {S[k].mean():.3f}±{S[k].std():.3f} | {d.mean():+.3f} (쌍별 std {d.std():.3f}) | {bi[k]:+.1f} | {bl[k]:+.1f} | {sig} |'
H='| 요소 | nCV 평균±std | Δ vs 기준 | hi(>230) 편향 | lo(<170) 편향 | |Δ|>기준 std? |\n|---|---|---|---|---|---|'
L=['# DAKER 요소별 순수 기여 (ablation_all)','','평가: outer 5-fold × 10회 (random_state 100+s), 하이퍼파라미터(RidgeCV 알파, 랭크매칭 α)와 NNLS 가중치는 outer train 안에서만 결정. 모델 OOF(미세조정 등)는 원래 5-fold(rs 42)로 만든 OOF를 그대로 쓰므로 이 부분은 2단 OOF 재사용(약한 낙관 가능).','',
 '## A) 단순 기준 모델(밝기+결정립 Ridge)에 하나씩 더하기','',H,row('A|base','A|base')]
A=sorted([k for k in K if k.startswith('A|+')],key=lambda k:S[k].mean())
L+=[row(k,'A|base') for k in A]
L+=['','## B) 최고 블렌드(BL1+ViT-B원본평균+ViT-B변환평균, 가중치 fold 내 NNLS 재추정)에서 하나씩 빼기','',H,row('B|최고블렌드','B|최고블렌드')]
L+=[row(k,'B|최고블렌드') for k in ['B|-BL1','B|-ViT-B 원본(3seed평균)','B|-ViT-B 변환(3seed평균)','B|+랭크매칭']]
L+=['','BL1 내부 특징군 빼기 (BL1을 fold 안에서 HX-Ridge+DINO임베딩-Ridge로 재구성, 기준 = 재구성 블렌드):','',H,row('B|BL1재구성+B원본+B변환','B|BL1재구성+B원본+B변환')]
L+=[row(k,'B|BL1재구성+B원본+B변환') for k in K if k.startswith('B|BL1재구성 −')]
L+=['','## C) 단독 점수와 잔차 상관','','잔차 = y − (최고블렌드 nCV OOF 10회 평균), 기준잔차 = y − (base Ridge). 상관이 |0.09| 넘으면 n=500에서 대략 p<0.05.','','| 요소 | 단독 RMSE | hi 편향 | lo 편향 | corr(예측, 최고블렌드 잔차) | corr(예측, base 잔차) |','|---|---|---|---|---|---|']
import sys;sys.argv=['x'];exec(open('./runs/ablation_all/loadM.py').read())
rows=[]
for k in [k for k in K if k.startswith('C|')]:
    p=np.mean([r[k] for r in R],0);rows.append((S[k].mean(),f'| {k[2:]} (특징 Ridge, nCV±{S[k].std():.3f}) | {S[k].mean():.3f} | {bi[k]:+.1f} | {bl[k]:+.1f} | {np.corrcoef(p,res)[0,1]:+.3f} | {np.corrcoef(p,resb)[0,1]:+.3f} |'))
for k,p in M.items(): rows.append((rm(p),f'| {k} (OOF) | {rm(p):.3f} | {(p-y)[hi].mean():+.1f} | {(p-y)[lo].mean():+.1f} | {np.corrcoef(p,res)[0,1]:+.3f} | {np.corrcoef(p,resb)[0,1]:+.3f} |'))
L+=[x for _,x in sorted(rows)]
L+=['','## 빠진 요소','- 구간 분류+회귀(bin cls, nCV 13.56 보고): 박스에 OOF 파일 없음(맥스튜디오 쪽 결과만 존재) → 제외.','- 쌍별 순위 손실(d4_rank), crop140, s448, dino5(마스크 채널): 맥스튜디오에서 아직 진행/미회수 → 제외.','- 랭크매칭은 OOF가 아니라 fold 안에서 직접 계산해 A·B에 포함.']
open('report.md','w').write('\n'.join(L));print('\n'.join(L))
