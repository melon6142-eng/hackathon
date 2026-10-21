# Exp A/B/C report

작성: 2026-10-13 09:30 CST (Asia/Shanghai)

기준 블렌드: BL1 + ViT-B orig 3시드 평균 + ViT-B trans 3시드 평균
- base nCV (10×5 NNLS+intercept) = **12.601** (레퍼런스 일치)
- base hi>230 bias = -27.2, lo<170 = 17.4

채택 기준: RMSE 한 번 0.05가 아니라 여러 fold/seed에서 샘플별 SSE 개선.
상관 ≥0.97이면 NNLS 사실상 제외. 개선 <0.05면 fine-tune 보류.
중복 금지: dino12 multilayer / grain_agree / fold_rs 검증과 동일 실험 재실행 없음.

## 실험 A — DINOv3 ViT-B/16 frozen

주력: CLS + 중간·최종층 patch mean/std (`A_clsmeanstd`), 4-way TTA, Ridge OOF (rs=42), 기존 BL1 블렌드와 nested NNLS.

| tag | solo OOF | hi bias | nCV(+base) | Δ | nCV hi | corr BL1/Bo/Bt | SSE win율 | NNLS w_new | 판정 |
|---|---|---|---|---|---|---|---|---|---|
| A_clsmeanstd | 13.425 | -30.8 | 12.606 | +0.006 | -27.3 | 0.923/0.877/0.873 | 0.16 | 0.000 | 보류/기각 |
| A_last_clsmean | 14.130 | -32.7 | 12.601 | +0.000 | -27.2 | 0.898/0.829/0.826 | 0.12 | 0.000 | 보류/기각 |
| A_last_all | 13.975 | -32.3 | 12.602 | +0.002 | -27.2 | 0.905/0.845/0.839 | 0.13 | 0.000 | 보류/기각 |
| A_mid_all | 13.236 | -29.9 | 12.625 | +0.024 | -27.4 | 0.899/0.864/0.866 | 0.30 | 0.150 | 보류/기각 |

- 주력 solo OOF=13.425 (DINOv2 frozen clsmean 14.06보다 약간 나음), 블렌드 nCV=12.606 (Δ+0.006)
- 상관=[0.9233062707768378, 0.8773731674576997, 0.8734006553935684], full NNLS w=['0.440', '0.283', '0.277', '0.000', '0.000']
- SSE win율=0.158 (0.5 미만 → 샘플별로도 이득 없음)
- 고경도 편향: solo -30.8 → 블렌드 후 -27.3 (base -27.2 대비 변화 없음)
- **판정: 보류/기각 — RMSE 개선 <0.05이고 NNLS 가중≈0 → DINOv3 fine-tune 보류**

## 실험 B — DINO patch 이웃 유사도 (분할 없음)

DINOv3 patch 토큰 좌우·상하·대각 cos/L2 분포 (층 2×방향 4×통계 5×메트릭 2 = 80维). grain_agree와 코드/폴더 미공유.

| tag | solo OOF | hi bias | nCV(+base) | Δ | nCV hi | corr BL1/Bo/Bt | SSE win율 | NNLS w_new | 판정 |
|---|---|---|---|---|---|---|---|---|---|
| B_neighbor | 15.297 | -35.2 | 12.601 | +0.001 | -27.2 | 0.771/0.694/0.701 | 0.10 | 0.000 | 보류/기각 |

- 블렌드 잔차↔B 잔차 상관=0.859 (새 신호 약함)
- 상관은 0.97 미만이나 NNLS w≈0, Δ≈0, SSE win=0.10
- **판정: 보류/기각 — GNN 확장 금지**. grain_agree 결과와만 비교.

## 실험 C — mild blur/noise 민감도 (진단)

clean frozen-DINOv2 Ridge OOF RMSE = 14.079

| 섭동 | mean\|Δpred\| | RMSE(Δ) | corr(clean) | OOF RMSE | ΔRMSE | hi \|Δ\| |
|---|---|---|---|---|---|---|
| blur_0.5 | 2.058 | 2.762 | 0.9754 | 14.241 | +0.163 | 1.750 |
| blur_1.0 | 4.143 | 5.286 | 0.8932 | 14.685 | +0.606 | 4.167 |
| blur_1.5 | 6.451 | 7.719 | 0.8215 | 15.488 | +1.410 | 7.701 |
| noise_0.01 | 3.818 | 4.568 | 0.9720 | 14.815 | +0.736 | 3.083 |
| noise_0.02 | 8.295 | 9.487 | 0.9089 | 16.843 | +2.764 | 7.507 |
| noise_0.05 | 9.635 | 11.528 | 0.8019 | 17.752 | +3.673 | 9.618 |
| blur1_noise0.02 | 10.146 | 11.884 | 0.7657 | 18.027 | +3.948 | 10.326 |

**판정:** 민감함 → consistency/증강 실험 근거 있음 (우선순위는 A/B 이득 여부에 종속)

해석: blur σ=0.5만으로도 mean|Δ|≈2.1·corr 0.975, σ=1.0에서 corr 0.89·ΔRMSE +0.6. noise 0.01도 |Δ|≈3.8. 표현이 고주파/노이즈에 민감 → consistency 손실·약한 증강은 **근거는 있으나**, A/B가 블렌드 이득을 못 준 상태에서 최우선은 아님. dino12 cropaug/multilayer 대기 결과와 겹치지 않게 숫자만 비교.

## 종합 다음 추천

1. **A:** DINOv3 fine-tune 보류 (블렌드 Δ≥0, w_new≈0)
2. **B:** GNN 확장 금지; grain_agree(분할 기반) 결과만 비교
3. **C:** 민감 → consistency/증강 후보는 근거 있음. 다만 A/B 실패로 우선순위 낮음. 기존 cropaug·multilayer 시드 결과를 먼저 볼 것
4. 대기 중 dino12 (multilayer s1/s2/trans/asym + fold_rs=7) · grain_agree와 **중복 실행 없이** 결과 비교만

산출물: `cnn/runs/exp_abc/` (feats_a.npz, feats_b.npz, diag_c.json, oof/, report.md, results.json)
