# 한국어 report.md: 모든 ablation OOF를 기존 P3/P2/DINO와 비교
import csv,os,numpy as np
from sklearn.metrics import average_precision_score
R='~/AI/daker';y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))]);hi=y>230;lo=y<170;mid=~hi&~lo
G='../gen_p2/'
NM={'dino':'frozen DINO (사전학습 없음)','pre':'P2 파라미터 회귀(단일 이미지 풀)','pp':'P3 전체(회귀+불변성, 100%)','reg':'(b) 쌍 데이터, 회귀만 (불변성 제거)','inv':'(b) 불변성만 (회귀 제거)','f30':'(a) P3, 합성 30%','f10':'(a) P3, 합성 10%','scr':'(c) 열화 생성기(위상 랜덤화, Voronoi 구조 파괴)'}
def src(k,rs):
    return {'dino':G+f'oof_frozen_dino_rs{rs}.npy','pre':G+f'oof_frozen_pre_rs{rs}.npy','pp':G+f'oof_frozen_pp_rs{rs}.npy'}.get(k,f'oof_{k}_rs{rs}.npy')
L=['# DAKER 연구 전용: Phase 3 이득 원인 분해 (one-change ablation)','','> 합성 데이터 사용 → **제출 불가, 연구용**. 모든 변형은 P3와 동일한 학습 스텝 수/백본/헤드/fold 풀, Ridge 평가는 frozen 8-TTA 특징 + fold-train RidgeCV.','']
for rs in (42,7):
    rows=[k for k in NM if os.path.exists(src(k,rs))]
    if len(rows)<=3 and rs!=42: continue
    L+=[f'## fold seed {rs}','','| 변형 | RMSE | Δ vs P3 | 고편향 | 저편향 | 고RMSE | 중간RMSE | 저RMSE | 고AUPRC | P@18 |','|---|---|---|---|---|---|---|---|---|---|']
    base=None
    for k in rows:
        o=np.load(src(k,rs));e=o-y;r=np.sqrt((e**2).mean())
        if k=='pp': base=r
    for k in rows:
        o=np.load(src(k,rs));e=o-y;r=np.sqrt((e**2).mean());top=np.argsort(-o)[:18]
        L.append(f'| {NM[k]} | {r:.3f} | {r-base:+.3f} | {e[hi].mean():+.1f} | {e[lo].mean():+.1f} | {np.sqrt((e[hi]**2).mean()):.1f} | {np.sqrt((e[mid]**2).mean()):.3f} | {np.sqrt((e[lo]**2).mean()):.3f} | {average_precision_score(hi,o):.3f} | {hi[top].mean():.3f} |')
    if base is not None and os.path.exists(src('dino',rs)):
        od=np.load(src('dino',rs));L+=['','DINO→변형 SSE 감소의 구간별 분해 (양수=개선, 전체 대비 비율):','','| 변형 | 총 ΔSSE | 고(18) | 저(38) | 중간(444) |','|---|---|---|---|---|']
        for k in rows:
            if k=='dino': continue
            d=(od-y)**2-(np.load(src(k,rs))-y)**2;T=d.sum()
            L.append(f'| {NM[k]} | {T:.0f} | {d[hi].sum():.0f} ({d[hi].sum()/T:.0%}) | {d[lo].sum():.0f} ({d[lo].sum()/T:.0%}) | {d[mid].sum():.0f} ({d[mid].sum()/T:.0%}) |')
    L.append('')
L+=['## 해석 가이드','- (a) 10/30%가 100%와 비슷 → 데이터 양 아님. (b) 회귀만 ≈ P3 → 파라미터 감독이 핵심, 불변성만 ≈ P3 → 불변성이 핵심. (c) 열화 생성기가 DINO 수준으로 떨어지면 → 미세조직 같은 구조(생성기 귀납 편향)가 이득의 원천.','- 참고: P3 원 목적에는 "구조 변화 예측" 항이 없음(회귀+코사인 불변성). 따라서 (b)는 회귀만/불변성만 두 변형으로 분해함.']
open('report.md','w').write('\n'.join(L)+'\n');print('\n'.join(L))
