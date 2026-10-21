#!/usr/bin/env python3
"""High-hardness pair DESIGN + candidate list only. No analysis execution."""
import os, json, csv, numpy as np
from sklearn.preprocessing import normalize
R='~/AI/daker'; O=f'{R}/cnn/runs/robust_pairs'
os.makedirs(O,exist_ok=True)
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
ids=[r['ID'] for r in csv.DictReader(open(f'{R}/open/train.csv'))]

# Use existing frozen DINOv2 last-layer CLS+mean from feat_dinoB (already extracted)
z=np.load(f'{R}/cnn/runs/feat_dinoB/feats.npz')
# last layer L1
E=np.concatenate([z['cls_L1'][:500], z['mean_L1'][:500]],1).astype(np.float64)
E=normalize(E)
hi=np.where(y>230)[0]
mid=np.where((y>=210)&(y<=230))[0]
print(f'hi={len(hi)} mid210-230={len(mid)}', flush=True)

# For each hi, nearest mid in DINO cosine space
cands=[]
for i in hi:
    sims=E[mid]@E[i]
    order=np.argsort(-sims)
    top=[]
    for rank,j in enumerate(order[:5]):
        mj=mid[j]
        top.append(dict(rank=rank+1, mid_id=ids[mj], mid_idx=int(mj), mid_y=float(y[mj]),
                        cos=float(sims[j]), dy=float(y[i]-y[mj])))
    cands.append(dict(hi_id=ids[i], hi_idx=int(i), hi_y=float(y[i]),
                      nearest=top, primary=top[0]))

# Design doc fields
design=dict(
    title='고경도(>230) ↔ 중간경도(210~230) DINO 근접 쌍 비교 — 설계만 (미실행)',
    status='design_only_no_analysis',
    pair_selection=dict(
        embedding='feat_dinoB cls_L1+mean_L1 (DINOv2 ViT-B/14 frozen, Llast), L2-normalized cosine',
        query='train 중 y>230 인 18장',
        pool='train 중 210<=y<=230',
        rule='각 query에 대해 pool에서 cosine 최근접 1개를 primary, top-5를 후보로 보관',
        exclusions='자기 자신 없음(교집합 공집합). test 이미지 사용 안 함',
        tie_break='cosine 동률 시 mid_idx 오름차순',
    ),
    metrics_to_compare=[
        '경도 차이 Δy = y_hi - y_mid',
        'DINO cosine / L2 거리',
        '밝기 단계 비율 (z<-1, -1~-0.5, -0.5~0.5, ≥0.5) 절대·상대',
        'dark-area ratio, largest dark blob area, coherence (기존 고경도 신호)',
        '결정립 크기/신장/방향 통계 (있다면 grain_* 특징)',
        '블렌드 OOF 예측 및 잔차 (p_hi, p_mid, residual)',
        '이웃 패치 유사도 분포 (exp B 특징, 있으면)',
        '픽셀 차분 |I_hi - I_mid| 의 공간 스펙트럼 (저주파 vs 고주파 에너지)',
    ],
    visualizations=[
        '18쌍 side-by-side 원본 + CLAHE (같은 gray window)',
        '쌍별 잔차 히트맵 (I_hi - warp/align? — 정렬 불가 시 단순 차분 + 히스토그램)',
        'DINO PCA/UMAP 2D에 hi(빨강)·mid(파랑)·pair edge',
        '특징 레이더: dark ratio, blob, coherence, brightness stages',
        '예측 잔차 막대: 각 쌍에서 모델이 hi를 mid처럼 보는 정도 (p_hi - y_mid)',
    ],
    success_fail_criteria=dict(
        note='고정 0.05 RMSE 컷이 아니라 재현·가설 검증 중심',
        success_hypothesis=[
            '가설H1: hi↔nearest-mid 쌍이 시각·측정특징에서 체계적으로 다르고, 그 차이가 경도 Δy와 단조성(Spearman)을 가짐',
            '가설H2: 차이가 DINO에 잘 안 잡히는 축(예: 대형 dark blob 위상)에 집중 → 표현 보강 타깃 명확',
            '재현: 임베딩을 Lmid / DINOv3 / HX로 바꿔도 상위 쌍의 ≥50%가 겹치거나, 같은 특징 축이 top-3에 반복',
        ],
        fail_if=[
            '최근접 mid가 사실상 랜덤(Δy·특징 차이가 null 분포와 구분 안 됨)',
            '쌍마다 다른 특징이 top이라 공통 실패 모드를 못 찾음',
            '이미 블렌드 잔차가 설명하는 축만 재발견 (잔차 상관 |ρ|>0.9 with existing dark/coherence)',
        ],
        decision_rule='성공 시 → 해당 축을 보조 특징으로 1회 nested 검증(새 백본 학습 없음). 실패 시 → 쌍 비교 경로 중단',
    ),
    reverse_problem_memo=dict(
        status='future_candidate_only_not_scheduled',
        idea='미세조직 생성 파라미터(신장비, 결정립 크기, 2상 분율, 기공, 노이즈/블러 플래그)를 이미지에서 역추정한 뒤 그 파라미터로 경도 회귀',
        why_later='gen/report에서 물리 공식 직접 피팅 CV 15.6+ 실패, 생성파라미터+morph를 BL1에 넣어도 0.2 이상 개선 없음. 역문제 자체가 새 표현 학습·합성 루프를 요구해 비용 큼',
        revisit_if='쌍 비교가 H2 성공(표현 밖 축)으로 나오고, 그 축이 생성 파라미터와 1:1 대응될 때',
    ),
    candidates=cands,
)
json.dump(design, open(f'{O}/pair_design.json','w'), indent=2, ensure_ascii=False)

# Human-readable candidate table
lines=['| hi_id | y_hi | mid_id | y_mid | Δy | cos |',
       '|---|---|---|---|---|---|']
for c in cands:
    p=c['primary']
    lines.append(f"| {c['hi_id']} | {c['hi_y']:.2f} | {p['mid_id']} | {p['mid_y']:.2f} | {p['dy']:.2f} | {p['cos']:.4f} |")
open(f'{O}/pair_candidates.md','w').write('\n'.join(lines)+'\n')
print('\n'.join(lines), flush=True)
print('WROTE pair_design.json pair_candidates.md', flush=True)
