# usage: python stage.py NAME "dinofile:layers:stats;dinofile:..." [hx]
# dinofile e.g. vitb14_252_L12ms ; layers like 9-12 (1-indexed of 12) ; stats m,s,ms
import os,sys,json,time
os.environ['OMP_NUM_THREADS']='3'; os.environ['OPENBLAS_NUM_THREADS']='3'
sys.path.insert(0,'lib'); import numpy as np,pandas as pd
from cv import y as gety
from hxfold import base,PolyTop
from blend2 import nested,fit_predict
OUT='./out'; ST='state_best.json'
name,dspec=sys.argv[1],sys.argv[2]; use_hx=len(sys.argv)>3
Y=gety()
def dino(sp,f,lay,st):
    F=np.load(f'dino/{f}_{sp}.npy',mmap_mode='r'); D=F.shape[1]//24
    a,b=map(int,lay.split('-')); ks=[0,1] if st=='ms' else ([0] if st=='m' else [1])
    return np.hstack([np.asarray(F[:,((L-1)*2+k)*D:((L-1)*2+k+1)*D]) for L in range(a,b+1) for k in ks])
Xs,Xt,sp=[],[],[]
for d in [x for x in dspec.split(';') if x]:
    f,lay,st=d.split(':'); Xs.append(dino('train',f,lay,st)); Xt.append(dino('test',f,lay,st)); sp.append(None)
if use_hx:
    Xb,nb,cand=base('train'); Xbt,_,_=base('test'); Xs.append(Xb); Xt.append(Xbt); sp.append(PolyTop(nb,cand))
o,rs,w=nested(Xs,sp,Y)
hi=Y>230; lo=Y<170
line=f'| {name} | {dspec} {"+HX(fold내 선택)" if use_hx else ""} | {rs.mean():.3f} (반복 {"/".join(f"{r:.2f}" for r in rs)}) | {np.mean(o[hi]-Y[hi]):+.1f} | {np.mean(o[lo]-Y[lo]):+.1f} | 가중 {np.round(w,2).tolist()} | {time.strftime("%H:%M")} 자동 |\n'
best=json.load(open(ST)) if os.path.exists(ST) else {'rmse':99,'reps':[99,99,99],'name':'none'}
imp= rs.mean() < best['rmse']-0.05 and all(rs < np.array(best['reps']))
line=line.replace('자동 |', '자동, **개선→제출파일 생성** |' if imp else f'자동, 미채택(현 최고 {best["name"]} {best["rmse"]:.3f}) |')
open(f'{OUT}/journal.md','a').write(line); print(line,flush=True)
if imp:
    p,_=fit_predict(Xs,sp,Y,Xt)
    ss=pd.read_csv('./open/sample_submission.csv'); ids=sorted(x[:-4] for x in os.listdir('./open/test'))
    s=pd.DataFrame({'ID':ids,'hardness':p}).set_index('ID').loc[ss.ID].reset_index()
    assert len(s)==1000 and s.hardness.notna().all() and set(s.ID)==set(ss.ID)
    fn=f'{OUT}/submission_{name}_cv{rs.mean():.2f}.csv'; s.to_csv(fn,index=False)
    json.dump({'rmse':float(rs.mean()),'reps':rs.tolist(),'name':name,'file':fn},open(ST,'w'))
    open(f'{OUT}/BEST.md','w').write(f'# 현재 최고 후보 (자동 갱신 {time.strftime("%m-%d %H:%M")})\n- **{fn}** — 엄밀 중첩 CV {rs.mean():.3f} (반복 {rs.round(3).tolist()}), 예상 Public ≈ {rs.mean()-0.9:.1f} (CV−Public≈0.9 경향)\n- 구성: {dspec} {"+ HX(fold 내부 상위14 선택+2차항)" if use_hx else ""}; 가중 {np.round(w,3).tolist()}\n- 이전 수동 최고: out/submission_blend1_cv12.95.csv (CV 12.95, HX 선택 낙관 약간 포함)\n- 재현: cd runs && ../venv/bin/python stage.py {name} "{dspec}"{" hx" if use_hx else ""}\n')
