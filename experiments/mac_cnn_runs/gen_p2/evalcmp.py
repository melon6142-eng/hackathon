import sys,os,numpy as np,csv
R='~/AI/daker';y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))]);hi=y>230;lo=y<170
rs=sys.argv[1] if len(sys.argv)>1 else '42'
if rs=='42': B={'s42':'../dinoB_orig','s1':'../dinoB_orig_s1','s2':'../dinoB_orig_s2'}
else: B={f's{s}':f'../indepfold/rs{rs}_orig_s{s}' for s in (42,1,2)}
N={f's{s}':f'ft/pre_rs{rs}_s{s}' for s in (42,1,2)}
def m(o): e=o-y;return np.sqrt((e**2).mean()),e[hi].mean(),e[lo].mean(),np.sqrt((e[hi]**2).mean())
out=[f'## fold rs{rs}','| 모델 | seed | RMSE | 고경도 편향 | 저경도 편향 | 고경도 RMSE |','|---|---|---|---|---|---|']
avg={}
for nm,D in (('baseline',B),('param-pretrain',N)):
    os_=[]
    for s,d in D.items():
        if not os.path.exists(f'{d}/oof.npy'): continue
        o=np.load(f'{d}/oof.npy');os_.append(o);out.append(f'| {nm} | {s} | '+' | '.join(f'{v:.3f}' for v in m(o))+' |')
    if os_: a=np.mean(os_,0);avg[nm]=a;out.append(f'| {nm} | {len(os_)}seed평균 | '+' | '.join(f'{v:.3f}' for v in m(a))+' |')
if len(avg)==2:
    d=(avg['param-pretrain']-y)**2-(avg['baseline']-y)**2
    out.append(f'\nΔSSE/sample: 전체 {d.mean():+.2f}, 고경도 {d[hi].mean():+.2f}, 저경도 {d[lo].mean():+.2f}, 중간 {d[~hi&~lo].mean():+.2f}; 개선 샘플 비율 {(d<0).mean():.2f}')
    for s in N:
        if os.path.exists(f'{N[s]}/oof.npy') and os.path.exists(f'{B[s]}/oof.npy'): pass
print('\n'.join(out));open(f'eval_rs{rs}.md','w').write('\n'.join(out)+'\n')
