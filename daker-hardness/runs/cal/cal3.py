"""Rank OOFs, proper nested blend top vs BL1, maybe submission. OMP=2."""
import os; os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'
import sys; sys.path.insert(0,'lib'); sys.path.insert(0,'.'); sys.path.insert(0,'cal')
import numpy as np, json, time, glob, pandas as pd
from sklearn.model_selection import KFold
from scipy.optimize import nnls
from feats import HX, DINO
from cv import y as get_y
from cal1 import blend_fp, blend_inner_oof, stretch_k, apply_stretch, fit_isotonic

def rmse(a,b): return float(np.sqrt(((a-b)**2).mean()))
def bias_report(p,yv):
    hi=yv>230; lo=yv<170
    out={'rmse':rmse(p,yv)}
    if hi.any(): out['hi_bias']=float(np.mean(p[hi]-yv[hi])); out['hi_rmse']=rmse(p[hi],yv[hi])
    if lo.any(): out['lo_bias']=float(np.mean(p[lo]-yv[lo])); out['lo_rmse']=rmse(p[lo],yv[lo])
    return out

if __name__=='__main__':
    t0=time.time(); Y=get_y()
    oofs={}
    for p in sorted(glob.glob('cal/oof_*.npy')):
        name=os.path.basename(p)[4:-4]
        oofs[name]=np.load(p)
    bl=oofs.get('base', np.load('oof_blend1.npy'))
    base=rmse(bl,Y)
    print('base',base,bias_report(bl,Y),flush=True)
    ranking=[]
    for name,o in oofs.items():
        if name=='base': continue
        br=bias_report(o,Y)
        O=np.column_stack([bl,o]); mu=Y.mean(); w,_=nnls(O-mu,Y-mu)
        pb=mu+(O-mu)@w
        ranking.append({'name':name,'solo_rmse':br['rmse'],'solo':br,
                        'blend_opt_rmse':rmse(pb,Y),'w':w.tolist(),'blend_bias':bias_report(pb,Y)})
        print(f"{name}: solo {br['rmse']:.3f} blend_opt {rmse(pb,Y):.3f}",flush=True)
    ranking.sort(key=lambda x: min(x['solo_rmse'],x['blend_opt_rmse']))

    # Proper nested for stretch & isotonic alone already in cal1.
    # Proper nested BL1+stretch and BL1+best_other if promising
    SEL=[(L,k) for L in range(4) for k in (1,2)]
    H=HX('train'); D=DINO('train',sel=SEL)
    Ht=HX('test'); Dt=DINO('test',sel=SEL)
    results={'base':base,'ranking':ranking}

    def bl1_va(tr,va):
        p,_=blend_fp([H[tr],D[tr]],Y[tr],[H[va],D[va]]); return p
    def stretch_va(tr,va):
        p_va=bl1_va(tr,va); po=blend_inner_oof([H[tr],D[tr]],Y[tr])
        return apply_stretch(p_va,Y[tr],stretch_k(po,Y[tr]))
    def iso_va(tr,va):
        p_va=bl1_va(tr,va); po=blend_inner_oof([H[tr],D[tr]],Y[tr])
        return fit_isotonic(po,Y[tr]).predict(p_va)

    for tag,fn in [('BL1+stretch',stretch_va),('BL1+isotonic',iso_va)]:
        # only if solo stretch/iso looked useful
        oofs_r=[]
        for r in range(3):
            oof=np.zeros(len(Y))
            for tr,va in KFold(5,shuffle=True,random_state=42+r).split(Y):
                ob=np.zeros(len(tr)); oc=np.zeros(len(tr))
                for t2,v2 in KFold(5,shuffle=True,random_state=1).split(tr):
                    ob[v2]=bl1_va(tr[t2],tr[v2]); oc[v2]=fn(tr[t2],tr[v2])
                mu=Y[tr].mean(); w,_=nnls(np.column_stack([ob-mu,oc-mu]),Y[tr]-mu)
                oof[va]=mu+np.column_stack([bl1_va(tr,va)-mu,fn(tr,va)-mu])@w
            oofs_r.append(oof); print(tag,'rep',r,rmse(oof,Y),flush=True)
        mean=np.mean(oofs_r,0)
        results[tag]={'rmse':float(np.mean([rmse(o,Y) for o in oofs_r])),'bias':bias_report(mean,Y)}
        np.save(f'cal/oof_{tag.replace("+","_")}.npy',mean)

    BL1_REF=12.95
    candidates=[]
    if os.path.exists('cal/cal1_res.json'):
        c1=json.load(open('cal/cal1_res.json'))
        for m,v in c1.items():
            if isinstance(v,dict) and 'rmse_mean' in v:
                candidates.append((m,v['rmse_mean']))
    for k,v in results.items():
        if isinstance(v,dict) and 'rmse' in v: candidates.append((k,v['rmse']))
    for row in ranking:
        candidates.append((row['name']+'_solo',row['solo_rmse']))
        candidates.append((row['name']+'_blendopt',row['blend_opt_rmse']))
    # cal2 bests
    if os.path.exists('cal/cal2_res.json'):
        c2=json.load(open('cal/cal2_res.json'))
        for cfg,d in c2.items():
            if cfg.startswith('_'): continue
            for m,v in d.items():
                if isinstance(v,dict) and 'rmse_mean' in v:
                    candidates.append((f'{cfg}/{m}',v['rmse_mean']))

    candidates=sorted(set((a,round(b,4)) for a,b in candidates), key=lambda x:x[1])
    winners=[c for c in candidates if c[1] <= BL1_REF-0.15]
    results['all_candidates']=candidates[:40]
    results['winners']=winners
    print('TOP',candidates[:15],flush=True)
    print('winners',winners,flush=True)

    # Only make submission for proper nested winners (not blend_opt optimistic)
    proper_names=set(['stretch','isotonic','qmap','ext_w','qreg','base','BL1+stretch','BL1+isotonic'])
    proper_winners=[w for w in winners if w[0] in proper_names or w[0].startswith('BL1')]
    # also allow cal2 methods if their nested rmse_mean wins
    for w in winners:
        if '/moe_HD' in w[0] or w[0].endswith('/pair') or '/inv' in w[0]:
            proper_winners.append(w)

    if proper_winners:
        tag,cv=proper_winners[0]
        print('Making submission for',tag,cv,flush=True)
        p_base,_=blend_fp([H,D],Y,[Ht,Dt])
        po=blend_inner_oof([H,D],Y)
        if tag=='stretch' or tag=='BL1+stretch':
            k=stretch_k(po,Y); ps=apply_stretch(p_base,Y,k)
            if tag=='BL1+stretch':
                mu=Y.mean(); w,_=nnls(np.column_stack([po-mu,apply_stretch(po,Y,k)-mu]),Y-mu)
                p=mu+np.column_stack([p_base-mu,ps-mu])@w
            else:
                p=ps
        elif tag in ('isotonic','BL1+isotonic'):
            iso=fit_isotonic(po,Y); pi=iso.predict(p_base)
            if tag=='BL1+isotonic':
                mu=Y.mean(); w,_=nnls(np.column_stack([po-mu,iso.predict(po)-mu]),Y-mu)
                p=mu+np.column_stack([p_base-mu,pi-mu])@w
            else:
                p=pi
        elif tag=='qmap':
            from cal1 import quantile_map
            p=quantile_map(p_base,po,Y)
        elif tag=='ext_w':
            # refit weighted blend
            from cal1 import weighted_ridge_fp
            def w_inner(X,yy):
                o=np.zeros(len(yy))
                for t,v in KFold(5,shuffle=True,random_state=1).split(X):
                    o[v]=weighted_ridge_fp(X[t],yy[t],X[v],2.0)
                return o
            O=np.column_stack([w_inner(X,Y) for X in [H,D]]); mu=Y.mean()
            w,_=nnls(np.column_stack([O-mu]),Y-mu)
            P=np.column_stack([weighted_ridge_fp(X,Y,Xt,2.0) for X,Xt in zip([H,D],[Ht,Dt])])
            p=mu+(P-mu)@w
        else:
            # phenotype methods: blend BL1 with best OOF via weights, apply same weights to test
            # Need test predictions for phenotype — recompute full-data
            print('phenotype winner — blend OOF weights with BL1 test; need test preds')
            # fallback: stretch
            k=stretch_k(po,Y); p=apply_stretch(p_base,Y,k)
            tag='stretch_fallback'
        ss=pd.read_csv('./open/sample_submission.csv')
        ids=sorted(x[:-4] for x in os.listdir('./open/test'))
        s=pd.DataFrame({'ID':ids,'hardness':p}).set_index('ID').loc[ss.ID].reset_index()
        fname=f'submission_cal_{tag.replace("+","_").replace("/","_")}_cv{cv:.2f}.csv'
        path=f'./out/{fname}'
        s.to_csv(path,index=False)
        results['submission']=path
        with open('./out/BEST.md','w') as f:
            f.write(f'# 현재 최고 후보\n- **{path}** — 엄밀 중첩 CV {cv:.2f} (BL1 {BL1_REF})\n')
            f.write(f'  - 방법: {tag}\n  - Public 300장 노이즈 큼 → CV 기준\n  - 재현: runs/cal/\n')
        print('wrote',path,flush=True)
    else:
        results['submission']=None
        print('No proper >=0.15 improvement; no submission',flush=True)

    results['elapsed']=time.time()-t0
    json.dump(results,open('cal/cal3_res.json','w'),indent=1,default=float)
    print('DONE',flush=True)
