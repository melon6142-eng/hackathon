# DAKER 합성 미세조직 경도 예측 해커톤 — 작업 기록

대회: [DAKER Synthetic Microstructure Property Prediction AI](https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai) (평가지표 RMSE)

> 이 저장소에는 **대회 데이터(train/test 이미지, train.csv 라벨, sample_submission), 특징·예측 배열, 제출 CSV, 모델 가중치가 포함되어 있지 않습니다.** 대회 규정에 따라 데이터는 대회 페이지에서 직접 받아야 합니다.

| 위치 | 내용 |
|---|---|
| [HISTORY.md](HISTORY.md) | 시간순 시도·변경·결과·채택/기각 이유 (한국어) |
| [final/](final/) | 규정준수 최종 파이프라인 — `final_clean/`(재구축 파이프라인, 누수 감사), `uni_none_final/`(최종 선택: 증강 없는 ViT 블렌드, decision.md / eval.md) |
| [experiments/](experiments/) | 실험별 스크립트와 report.md. `RESEARCH_ONLY` 표시 실험은 새 합성 이미지 사전학습을 써서 대회 제출 불가 |
| [reports/](reports/) | PDF 보고서 (10/15 v1~v3, 10/17 균형샘플링·D4) |

## 재현 (요약)
1. 대회 데이터를 받아 `final/final_clean/` 기준 상대경로에 배치하고, 스크립트 안의 데이터 경로를 맞춥니다.
2. 환경: Python venv — torch 2.14.1, timm 1.0.30, scikit-learn 1.9.1, numpy 2.5.4, opencv 5.0.0, scikit-image 0.26.0, PyWavelets. macOS(Apple Silicon, MPS)에서 실행. `HF_HUB_OFFLINE=1`.
3. 사전학습 가중치: Meta torch.hub `dinov2_vitb14`(Apache-2.0) — BL1 특징, timm `vit_base_patch14_dinov2.lvd142m`(HF, 2023-05-09) — ViT 미세조정. sha256은 `final/final_clean/README.md`.
4. `final/final_clean/run_all.sh all` → BL1 + ViT 6개 + 블렌드. 최종 선택은 ViT를 증강 없는(uni_none) 설정으로 학습한 eq3 블렌드 (`final/uni_none_final/`).

자세한 단계·누수 수정 내역은 [final/final_clean/README.md](final/final_clean/README.md).

## 기타 폴더
- `experiments/mac_cnn_runs/` — Mac Studio에서 돌린 ViT/CNN 실험 스크립트·보고서 (gen_p2, p3_ablate, ssl_train은 RESEARCH-ONLY).
- `daker-hardness/` — 10/12 최초 커밋의 초기 탐색 코드 (히스토리 재작성으로 데이터 파생 이미지·CSV·JSON 제거됨).
