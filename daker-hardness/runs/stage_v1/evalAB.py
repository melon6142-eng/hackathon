# 엄밀 중첩 CV(5x3) 비교. usage: python evalAB.py
import os; os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'
import sys; H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,H+'/../lib'); sys.path.insert(0,H); os.chdir(H+'/..')
import numpy as np,pandas as pd,json,time
from cv import y as gety; from hxfold import base,PolyTop; from blend2 import nested,fit_predict; from blend import fit_ridge
from featB import PatternDist
Y=gety(); OUT='./out'
A=pd.read_csv(H+'/A_train.csv').drop(columns='ID'); At=pd.read_csv(H+'/A_test.csv').drop(columns='ID')
D=768; sel=[(L,k) for L in range(4) for k in (1,2)]
dino=lambda sp: np.hstack([np.load(f'dino/vitb14_252_{sp}.npy',mmap_mode='r')[:,(L*4+k)*D:(L*4+k+1)*D] for L,k in sel]).astype(np.float32)
Xd,Xdt=dino('train'),dino('test'); Xb,nb,cand=base('train'); Xbt,_,_=base('test')
IB=np.arange(500)[:,None].astype(float); IBt=(500+np.arange(1000))[:,None].astype(float)
comp={'A':(A.values,At.values,None),'B':(IB,IBt,PatternDist()),'HX':(Xb,Xbt,PolyTop(nb,cand)),'DINO':(Xd,Xdt,None)}
exps=[('A단독',['A']),('B단독',['B']),('BL1(HX fold내+DINO)',['HX','DINO']),('BL1+A',['HX','DINO','A']),('BL1+B',['HX','DINO','B']),('BL1+A+B',['HX','DINO','A','B'])]
res={}
log=open(H+'/results.md','a')
for nm,cs in exps:
    t=time.time(); o,rs,w=nested([comp[c][0] for c in cs],[comp[c][2] for c in cs],Y); res[nm]=(o,rs,w,cs)
    hi=Y>230; lo=Y<170
    line=f'| {nm} | {rs.mean():.3f} | {"/".join(f"{r:.2f}" for r in rs)} | {np.mean(o[hi]-Y[hi]):+.1f} | {np.mean(o[lo]-Y[lo]):+.1f} | {dict(zip(cs,np.round(w,2).tolist()))} | {(time.time()-t)/60:.0f}분 |'
    print(line,flush=True); log.write(line+'\n'); log.flush(); np.save(H+f'/oof_{nm}.npy',o)
json.dump({k:{'rmse':float(v[1].mean()),'reps':v[1].tolist()} for k,v in res.items()},open(H+'/results.json','w'),ensure_ascii=False)
# 중요도 상위 10: 전체 train 표준화 Ridge 계수 |β| (+ 단변량 상관)
def top10(X,names):
    m=fit_ridge(X,Y); b=m[-1].coef_; c=[np.corrcoef(X[:,j],Y)[0,1] if X[:,j].std()>0 else 0 for j in range(X.shape[1])]
    i=np.argsort(-np.abs(b))[:10]; return [(names[j],round(float(b[j]),2),round(float(c[j]),2)) for j in i]
pdist=PatternDist().fit(IB,Y)
with open(H+'/top10.md','w') as f:
    f.write('## (A) 상위10 (표준화 Ridge 계수, 단변량 상관)\n'+'\n'.join(map(str,top10(A.values,list(A.columns))))+'\n\n## (B) 상위10 (train은 내부 OOF 특징)\n'+'\n'.join(map(str,top10(pdist.F,pdist.names)))+'\n')
# 유의 개선 시 제출파일
best=json.load(open('state_best.json')); bl=res['BL1(HX fold내+DINO)'][1]
for nm in ['BL1+A','BL1+B','BL1+A+B']:
    rs=res[nm][1]
    if rs.mean()<min(best['rmse'],bl.mean())-0.05 and (rs<bl).all():
        cs=res[nm][3]; p,w=fit_predict([comp[c][0] for c in cs],[comp[c][2] for c in cs],Y,[comp[c][1] for c in cs])
        ss=pd.read_csv('./open/sample_submission.csv'); ids=sorted(x[:-4] for x in os.listdir('./open/test'))
        s=pd.DataFrame({'ID':ids,'hardness':p}).set_index('ID').loc[ss.ID].reset_index(); assert len(s)==1000 and s.hardness.notna().all()
        fn=f'{OUT}/submission_stage_v1_{nm.replace("+","_")}_cv{rs.mean():.2f}.csv'; s.to_csv(fn,index=False)
        best={'rmse':float(rs.mean()),'reps':rs.tolist(),'name':nm,'file':fn}; json.dump(best,open('state_best.json','w'))
        open(f'{OUT}/BEST.md','w').write(f'# 현재 최고 후보 (갱신 {time.strftime("%m-%d %H:%M")})\n- **{fn}** — 엄밀 중첩 CV {rs.mean():.3f} (반복 {np.round(rs,3).tolist()}), BL1 동일조건 {bl.mean():.3f}. 예상 Public ≈ {rs.mean()-0.9:.1f}\n- 구성: {cs}, 가중 {np.round(w,3).tolist()} (HX 상호작용 선택은 fold 내부)\n- 재현: runs/stage_v1/evalAB.py\n- 이전: out/submission_blend1_cv12.95.csv\n'); print('NEW BEST',fn,flush=True)
print('ALL DONE',flush=True)
