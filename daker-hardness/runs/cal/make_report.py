import json,os,glob,numpy as np,pandas as pd
from cv import y as get_y
import sys; sys.path.insert(0,'lib'); sys.path.insert(0,'.')
Y=get_y()
def rmse(a,b): return float(np.sqrt(((a-b)**2).mean()))
def bias(p):
    hi=Y>230; lo=Y<170
    return rmse(p,Y), float(np.mean(p[hi]-Y[hi])), float(np.mean(p[lo]-Y[lo]))

lines=[]
lines.append('# DAKER 양 끝 보정·양상 역산 실험 보고\n')
lines.append('기준 BL1 OOF RMSE ≈ 12.95 (엄밀 중첩 시 ~13.03). Public 300장은 노이즈 큼 → CV로 판단.\n')
lines.append('폴더 `runs/cal/`, 스레드 2, 모든 평가는 중첩 반복 CV.\n')

if os.path.exists('cal/cal1_res.json'):
    c1=json.load(open('cal/cal1_res.json'))
    lines.append('\n## 1) 양 끝 보정\n')
    lines.append('| 방법 | CV RMSE | hi(>230) bias | lo(<170) bias |\n|---|---:|---:|---:|\n')
    for m in ['base','stretch','isotonic','qmap','ext_w','qreg']:
        if m not in c1: continue
        b=c1[m]['oof_bias']
        lines.append(f"| {m} | {c1[m]['rmse_mean']:.3f} | {b.get('hi_bias')} | {b.get('lo_bias')} |\n")
else:
    lines.append('\n## 1) 양 끝 보정 — 미완료\n')

if os.path.exists('cal/cal2_res.json'):
    c2=json.load(open('cal/cal2_res.json'))
    lines.append('\n## 2) 양상별 역산·MoE·쌍비교\n')
    rows=[]
    for cfg,d in c2.items():
        if cfg.startswith('_'): continue
        for m,v in d.items():
            if isinstance(v,dict) and 'rmse_mean' in v:
                bi=v.get('bias') or v.get('oof_bias') or {}
                rows.append((v['rmse_mean'],cfg,m,bi.get('hi_bias'),bi.get('lo_bias')))
    rows.sort()
    lines.append('| RMSE | 설정 | 방법 | hi bias | lo bias |\n|---:|---|---|---:|---:|\n')
    for r,cfg,m,hi,lo in rows[:25]:
        lines.append(f'| {r:.3f} | {cfg} | {m} | {hi} | {lo} |\n')
else:
    lines.append('\n## 2) 양상 — 미완료\n')

if os.path.exists('cal/cal3_res.json'):
    c3=json.load(open('cal/cal3_res.json'))
    lines.append('\n## 3) BL1 블렌드·제출 결정\n')
    lines.append(f"- candidates: {c3.get('all_candidates')}\n")
    lines.append(f"- winners (≥0.15 개선): {c3.get('winners')}\n")
    lines.append(f"- submission: {c3.get('submission')}\n")
else:
    lines.append('\n## 3) 블렌드 — 미완료\n')

lines.append('\n## 결론\n')
lines.append('- Public 11.77은 300장 표본 운+꼬리 오차 집중 영향이 큼. CV 개선이 본선 지표.\n')
lines.append('- 양 끝 축소가 주 병목 → stretch/isotonic/분위매핑이 1순위 가설.\n')
lines.append('- 양상 묶음은 공변량 이동이 없을 때 이득이 제한적일 수 있음. MoE·역산·쌍비교 결과를 위를 참고.\n')
open('cal/report.md','w').write(''.join(lines))
print('wrote cal/report.md')
