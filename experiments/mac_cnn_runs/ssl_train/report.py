import json,glob,os,numpy as np
L=['# DAKER 규정준수 트랙: train 500장 VICReg 자기지도 사전학습 (합성·외부·test 학습 없음)','',
'- 방법: 외부 fold마다 학습 fold 이미지만으로 DINOv2 ViT-B(원본 lvd142m) 마지막 블록 + projector를 VICReg로 학습. 두 뷰 = 같은 train 이미지의 D4 회전/반전 + 감마·대비·밝기·약한 blur·노이즈. 검증 이미지는 사전학습에 들어가지 않음.',
'- 평가: 8-TTA frozen 특징(CLS+patch mean) + RidgeCV (Phase 3와 동일). eq3 = BL1+ViT orig3+ViT trans3 균등, eq4 = eq3 + SSL ridge 균등.',
'- 변형: v1 = 8ep lr3e-5 블록4개 / v2 = 30ep lr5e-5 블록6개 / v3 = 30ep lr1e-4 블록4개','']
S={}
for f in sorted(glob.glob('ev_*_rs*.json')):
    t,rs=f[3:-5].rsplit('_rs',1);S.setdefault(t,{})[rs]=json.load(open(f))
for t,D in S.items():
    L.append(f'## {t}');L.append('| seed | DINO ridge | SSL ridge | eq3 | eq4(+SSL) | Δ | SSL 고편향 | SSL 저편향 | eq4 AUPRC | eq4 P@18 | eq4 고편향 |');L.append('|'+'---|'*11)
    for rs,r in sorted(D.items(),key=lambda x:int(x[0])):
        L.append(f"| {rs} | {r['frozen DINO']['rmse']:.3f} | {r['SSL frozen']['rmse']:.3f} | {r['eq3']['rmse']:.3f} | {r['eq4(+SSL)']['rmse']:.3f} | {r['eq4(+SSL)']['rmse']-r['eq3']['rmse']:+.3f} | {r['SSL frozen']['hb']:+.1f} | {r['SSL frozen']['lb']:+.1f} | {r['eq4(+SSL)']['auprc']:.3f} | {r['eq4(+SSL)']['p18']:.3f} | {r['eq4(+SSL)']['hb']:+.1f} |")
    if len(D)>1:
        m=lambda k:np.mean([r[k]['rmse'] for r in D.values()]);L.append(f"\n{len(D)}seed 평균: DINO {m('frozen DINO'):.3f} / SSL {m('SSL frozen'):.3f} / eq3 {m('eq3'):.3f} / eq4 {m('eq4(+SSL)'):.3f}")
    L.append('')
if os.path.exists('verdict.txt'): L+=['## 판정',open('verdict.txt').read()]
open('report.md','w').write('\n'.join(L)+'\n')
