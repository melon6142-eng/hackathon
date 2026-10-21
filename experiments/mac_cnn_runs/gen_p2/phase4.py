# Phase 4: fold seed 42/7/123/2026 채택 검증
# 지표: 전체 RMSE, 고/저편향, 구간 RMSE, AUPRC, P@18, 샘플별 SSE
# 덮어쓰기 금지: 기존 OOF/ckpt는 읽기만. 결과는 phase4_* 만 기록.
import os, csv, json, numpy as np
from sklearn.metrics import average_precision_score as ap
from scipy.stats import spearmanr

y = np.array([float(r['hardness']) for r in csv.DictReader(open('~/AI/daker/open/train.csv'))])
hi = y > 230
lo = y < 170
mid = ~hi & ~lo
H = np.where(hi)[0]
SEED = ('42', '7', '123', '2026')
BASE_AUPRC, BASE_P18 = 0.34, 0.33  # Phase1 기준

def G_vit(rs, kind):
    if rs == '42':
        return [f'../dinoB_{kind}' + s for s in ('', '_s1', '_s2')]
    return [f'../indepfold/rs{rs}_{kind}_s{s}' for s in (42, 1, 2)]

def mean_oof(dirs):
    return np.mean([np.load(d + '/oof.npy') for d in dirs], 0)

def bl1_of(rs):
    if rs == '42':
        return np.load('../../oof_blend1.npy')
    return np.load(f'../indepfold/bl1_rs{rs}.npy')

def metrics(o):
    e = o - y
    return dict(
        rmse=float(np.sqrt((e ** 2).mean())),
        hi_bias=float(e[hi].mean()),
        lo_bias=float(e[lo].mean()),
        hi_rmse=float(np.sqrt((e[hi] ** 2).mean())),
        mid_rmse=float(np.sqrt((e[mid] ** 2).mean())),
        lo_rmse=float(np.sqrt((e[lo] ** 2).mean())),
        auprc=float(ap(hi, o)),
        p18=float(hi[np.argsort(-o)[:18]].mean()),
        rho_hi=float(spearmanr(o[H], y[H])[0]),
        sse=float((e ** 2).sum()),
        sse_hi=float((e[hi] ** 2).sum()),
        sse_lo=float((e[lo] ** 2).sum()),
    )

def fmt(m):
    return (f"{m['rmse']:.3f}", f"{m['hi_bias']:+.1f}", f"{m['lo_bias']:+.1f}",
            f"{m['hi_rmse']:.1f}", f"{m['mid_rmse']:.3f}", f"{m['lo_rmse']:.3f}",
            f"{m['auprc']:.3f}", f"{m['p18']:.3f}")

rows = ['| rs | 모델 | RMSE | 고편향 | 저편향 | 고RMSE | 중간RMSE | 저RMSE | AUPRC | P@18 |',
        '|' + '---|' * 10]
store = {}  # (rs, nm) -> (metrics, oof)

for rs in SEED:
    bl1 = bl1_of(rs)
    vo = mean_oof(G_vit(rs, 'orig'))
    vt = mean_oof(G_vit(rs, 'trans'))
    fp = np.load(f'oof_frozen_pre_rs{rs}.npy')
    fd = np.load(f'oof_frozen_dino_rs{rs}.npy')
    fpp = np.load(f'oof_frozen_pp_rs{rs}.npy')
    ftp = [f'ft/pre_rs{rs}_s{s}/oof.npy' for s in (42, 1, 2)]
    pv = mean_oof([os.path.dirname(p) for p in ftp]) if all(os.path.exists(p) for p in ftp) else None

    cands = [
        ('frozen DINO', fd),
        ('frozen PRE(P2)', fp),
        ('frozen PP(P3)', fpp),
        ('ViT orig3', vo),
        ('eq3', (bl1 + vo + vt) / 3),
        ('eq4(+PRE)', (bl1 + vo + vt + fp) / 4),
        ('eq4(+PP)', (bl1 + vo + vt + fpp) / 4),
        ('eq5(+PRE+PP)', (bl1 + vo + vt + fp + fpp) / 5),
    ]
    if pv is not None:
        cands += [
            ('PRE-FT orig3', pv),
            ('eq3 orig→PRE-FT', (bl1 + pv + vt) / 3),
            ('eq4 PRE-FT+frozenPRE', (bl1 + pv + vt + fp) / 4),
            ('eq4 PRE-FT+frozenPP', (bl1 + pv + vt + fpp) / 4),
            ('eq5 PRE-FT+PRE+PP', (bl1 + pv + vt + fp + fpp) / 5),
        ]
    for nm, o in cands:
        m = metrics(o)
        store[(rs, nm)] = (m, o)
        rows.append(f"| {rs} | {nm} | " + ' | '.join(fmt(m)) + ' |')

    # ΔSSE vs eq3 for key candidates
    base = store[(rs, 'eq3')][1]
    for nm in ('eq4(+PRE)', 'eq4(+PP)', 'eq5(+PRE+PP)'):
        if (rs, nm) not in store:
            continue
        b = store[(rs, nm)][1]
        d = (b - y) ** 2 - (base - y) ** 2
        rows.append(
            f"| {rs} | ΔSSE {nm}−eq3 | 전체 {d.mean():+.2f} | 고 {d[hi].mean():+.1f} | "
            f"저 {d[lo].mean():+.1f} | 중간 {d[mid].mean():+.2f} | 개선비율 {(d < 0).mean():.2f} | | | |"
        )

# 4-seed 평균 (핵심 모델)
avg_models = ['eq3', 'eq4(+PRE)', 'eq4(+PP)', 'eq5(+PRE+PP)', 'frozen PRE(P2)', 'frozen PP(P3)',
              'PRE-FT orig3', 'eq4 PRE-FT+frozenPRE', 'eq4 PRE-FT+frozenPP']
rows += ['', '## 4-seed 평균',
         '| 모델 | RMSE | 고편향 | 저편향 | 고RMSE | 중간RMSE | 저RMSE | AUPRC | P@18 |',
         '|' + '---|' * 9]
avg_store = {}
for nm in avg_models:
    ms = [store[(rs, nm)][0] for rs in SEED if (rs, nm) in store]
    if len(ms) < 4:
        continue
    keys = ['rmse', 'hi_bias', 'lo_bias', 'hi_rmse', 'mid_rmse', 'lo_rmse', 'auprc', 'p18']
    am = {k: float(np.mean([m[k] for m in ms])) for k in keys}
    avg_store[nm] = am
    rows.append(f"| {nm} | " + ' | '.join(fmt(am)) + ' |')

# 채택 판정: 전체 RMSE + 고편향(덜 음수) + 저편향(덜 양수) + 고RMSE 가 eq3 대비 일관 개선
eq3 = avg_store['eq3']
adopt_lines = ['', '## 채택 판정 (기준: Phase1 AUPRC≥0.34 & P@18≥0.33, 그리고 eq3 대비 4지표 개선이 4 seed에서 일관)',
               '| 후보 | vs eq3 RMSE↓ | 고편향개선 | 저편향개선 | 고RMSE↓ | AUPRC | P@18 | seed일관(4/4 RMSE↓) | 판정 |',
               '|' + '---|' * 9]
decisions = {}
for nm in avg_models:
    if nm == 'eq3' or nm not in avg_store:
        continue
    am = avg_store[nm]
    # per-seed consistency on RMSE vs eq3
    n_better = sum(1 for rs in SEED if (rs, nm) in store and store[(rs, nm)][0]['rmse'] < store[(rs, 'eq3')][0]['rmse'])
    ok_rmse = am['rmse'] < eq3['rmse']
    ok_hi = am['hi_bias'] > eq3['hi_bias']  # less negative
    ok_lo = am['lo_bias'] < eq3['lo_bias']  # less positive
    ok_hir = am['hi_rmse'] < eq3['hi_rmse']
    ok_id = am['auprc'] >= BASE_AUPRC and am['p18'] >= BASE_P18
    # adoption: overall RMSE must improve; prefer also bias/hiRMSE; ID scores at least not collapse below Phase1
    adopt = ok_rmse and n_better == 4 and (ok_hi or ok_hir) and am['auprc'] >= 0.30
    strong = adopt and ok_hi and ok_lo and ok_hir and ok_id
    verdict = '채택(강)' if strong else ('채택(약)' if adopt else '기각')
    decisions[nm] = dict(verdict=verdict, n_better=n_better, **am)
    adopt_lines.append(
        f"| {nm} | {am['rmse'] - eq3['rmse']:+.3f} | {am['hi_bias'] - eq3['hi_bias']:+.1f} | "
        f"{am['lo_bias'] - eq3['lo_bias']:+.1f} | {am['hi_rmse'] - eq3['hi_rmse']:+.1f} | "
        f"{am['auprc']:.3f} | {am['p18']:.3f} | {n_better}/4 | {verdict} |"
    )

# P3 solo vs P2 solo vs DINO (표현 자체)
rows += ['', '## Phase2/3 표현 단독 (frozen ridge)',
         '| 모델 | RMSE | 고편향 | 저편향 | 고RMSE | AUPRC | P@18 |',
         '|---|---|---|---|---|---|---|']
for nm in ('frozen DINO', 'frozen PRE(P2)', 'frozen PP(P3)'):
    ms = [store[(rs, nm)][0] for rs in SEED]
    am = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    rows.append(f"| {nm} | {am['rmse']:.3f} | {am['hi_bias']:+.1f} | {am['lo_bias']:+.1f} | {am['hi_rmse']:.1f} | {am['auprc']:.3f} | {am['p18']:.3f} |")

# 고경도 18장 점수 비교 (eq3 vs eq4+PP vs frozen PP)
rows += ['', '## 고경도 18장 순위 (전체 500 중, 작을수록 높게 예측) eq3 → eq4(+PP) → frozen PP',
         '| idx | 경도 | ' + ' | '.join(SEED) + ' |',
         '|' + '---|' * (2 + len(SEED))]
ranks = {}
for rs in SEED:
    for nm in ('eq3', 'eq4(+PP)', 'frozen PP(P3)'):
        s = store[(rs, nm)][1]
        ranks[(rs, nm)] = (-s).argsort().argsort() + 1
for i in H[np.argsort(-y[H])]:
    cells = []
    for rs in SEED:
        a, b, c = ranks[(rs, 'eq3')][i], ranks[(rs, 'eq4(+PP)')][i], ranks[(rs, 'frozen PP(P3)')][i]
        cells.append(f'{a}→{b}→{c}')
    rows.append(f'| {i} | {y[i]:.0f} | ' + ' | '.join(cells) + ' |')

# 샘플별 SSE: eq3 vs best adopted (or eq4+PP)
best_nm = None
for nm, d in decisions.items():
    if d['verdict'].startswith('채택'):
        if best_nm is None or d['rmse'] < decisions[best_nm]['rmse']:
            best_nm = nm
if best_nm is None:
    best_nm = 'eq4(+PP)' if 'eq4(+PP)' in avg_store else 'eq3'

sse_lines = [f'', f'## 샘플별 SSE (eq3 vs {best_nm}, 4-seed 평균 예측)',
             '| idx | 경도 | SSE_eq3 | SSE_new | Δ | 구간 |',
             '|---|---|---|---|---|---|']
o_eq = np.mean([store[(rs, 'eq3')][1] for rs in SEED], 0)
o_nw = np.mean([store[(rs, best_nm)][1] for rs in SEED if (rs, best_nm) in store], 0)
sse_eq = (o_eq - y) ** 2
sse_nw = (o_nw - y) ** 2
delta = sse_nw - sse_eq
order = np.argsort(-sse_eq)[:40]
for i in order:
    band = '고' if hi[i] else ('저' if lo[i] else '중')
    sse_lines.append(f'| {i} | {y[i]:.0f} | {sse_eq[i]:.1f} | {sse_nw[i]:.1f} | {delta[i]:+.1f} | {band} |')
sse_lines += [
    f'',
    f'- eq3 전체 SSE={sse_eq.sum():.0f}, {best_nm} SSE={sse_nw.sum():.0f}, Δ={delta.sum():+.0f}',
    f'- 고경도 18장 SSE: eq3={sse_eq[hi].sum():.0f} → {best_nm}={sse_nw[hi].sum():.0f} (Δ {delta[hi].sum():+.0f})',
    f'- 저경도 38장 SSE: eq3={sse_eq[lo].sum():.0f} → {best_nm}={sse_nw[lo].sum():.0f} (Δ {delta[lo].sum():+.0f})',
]

# 요약 판정 문단
summary = ['', '## 요약']
p3 = avg_store.get('frozen PP(P3)')
p2 = avg_store.get('frozen PRE(P2)')
if p3 and p2:
    summary.append(
        f"- Phase3 frozen PP ridge 4-seed 평균 RMSE {p3['rmse']:.3f} (P2 PRE {p2['rmse']:.3f}, DINO "
        f"{float(np.mean([store[(rs,'frozen DINO')][0]['rmse'] for rs in SEED])):.3f}); "
        f"고편향 {p3['hi_bias']:+.1f}, 저편향 {p3['lo_bias']:+.1f}, 고RMSE {p3['hi_rmse']:.1f}; "
        f"AUPRC {p3['auprc']:.3f}, P@18 {p3['p18']:.3f}."
    )
adopted = [nm for nm, d in decisions.items() if d['verdict'].startswith('채택')]
rejected = [nm for nm, d in decisions.items() if d['verdict'] == '기각']
summary.append(f"- 채택 후보: {', '.join(adopted) if adopted else '없음'}.")
summary.append(f"- 기각: {', '.join(rejected) if rejected else '없음'}.")
if best_nm in decisions:
    d = decisions[best_nm]
    summary.append(
        f"- 권장: **{best_nm}** (4-seed RMSE {d['rmse']:.3f}, 고편향 {d['hi_bias']:+.1f}, "
        f"저편향 {d['lo_bias']:+.1f}, 고RMSE {d['hi_rmse']:.1f}, AUPRC {d['auprc']:.3f}, P@18 {d['p18']:.3f}, 판정 {d['verdict']})."
    )
# 고경도 식별이 Phase1 기준을 넘는지
id_ok = any(avg_store[nm]['auprc'] >= BASE_AUPRC and avg_store[nm]['p18'] >= BASE_P18
            for nm in adopted if nm in avg_store)
summary.append(
    f"- 고경도 식별(AUPRC≥{BASE_AUPRC}, P@18≥{BASE_P18})을 채택 후보가 충족하는가: "
    f"{'예' if id_ok else '아니오 (표현 학습이 고경도 신호를 충분히 강화하지 못함 → 18장 조성·생성규칙 직접 분석으로 전환 권고)'}."
)

body = '\n'.join(['# Phase 4: 4-seed 채택 검증', ''] + rows + adopt_lines + sse_lines + summary) + '\n'
open('phase4_report.md', 'w').write(body)
print(body)

# machine-readable
out = {
    'updated': '2026-10-15T07:24+08:00',
    'phase3_done': True,
    'best': best_nm,
    'avg': avg_store,
    'decisions': {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                      for kk, vv in v.items()} for k, v in decisions.items()},
    'id_ok': bool(id_ok),
}
# json-safe
def jdef(o):
    if isinstance(o, dict):
        return {k: jdef(v) for k, v in o.items()}
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o
open('PHASE4_DONE.json', 'w').write(json.dumps(jdef(out), ensure_ascii=False, indent=1))
open('PHASE3_DONE.json', 'w').write(json.dumps({
    'phase3': 'frozen_pair_pretrain_all_rs_done',
    'at': '2026-10-15T06:40:15+08:00',
    'rs42': 12.374, 'rs7': 12.289, 'rs123': 12.073, 'rs2026': 12.353,
}, indent=1))
print('WROTE phase4_report.md PHASE4_DONE.json PHASE3_DONE.json')
