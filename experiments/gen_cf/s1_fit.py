# Step1-2: 생성기 파라미터를 실제 train 500 분포에 맞추고 현실성 평가 (test 미사용)
import sys,os,json,time,numpy as np,torch,csv
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from gen import gen,sample_prior,clip,PRIOR,KEYS
from multiprocessing import Pool
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict,StratifiedKFold
from sklearn.metrics import roc_auc_score
O=os.path.dirname(os.path.abspath(__file__))
def g1(a): return gen(*a)
def G(plist,seeds,pool): return np.stack(pool.map(g1,list(zip(plist,seeds)),chunksize=8))
fy=np.fft.fftfreq(256)[:,None];fx=np.fft.fftfreq(256)[None,:];FR=np.sqrt(fx**2+fy**2);EDG=np.geomspace(1/256,0.5,13)
def stats(X):
    out=[]
    for x in X:
        P=np.abs(np.fft.fft2(x-x.mean()))**2
        ps=[np.log(P[(FR>=EDG[j])&(FR<EDG[j+1])].mean()+1e-12) for j in range(12)]
        q=np.percentile(x,[1,5,25,50,75,95,99])
        out.append(np.r_[x.mean(),x.std(),q,ps])
    return np.array(out)
SN=['mean','std']+[f'q{q}' for q in (1,5,25,50,75,95,99)]+[f'psd{j}' for j in range(12)]
def auc2(A,B,seed=0):
    X=np.r_[A,B];y=np.r_[np.zeros(len(A)),np.ones(len(B))]
    p=cross_val_predict(LogisticRegression(C=0.1,max_iter=3000),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=seed),method='predict_proba')[:,1]
    return roc_auc_score(y,p)
if __name__=='__main__':
    import emb
    from scipy.spatial import cKDTree
    pool=Pool(10);L=open(f'{O}/s1.log','a')
    def P_(s): print(s,flush=True);L.write(s+'\n');L.flush()
    R='~/AI/daker';Xr,_=torch.load(f'{R}/cnn/imgs.pt');Xr=Xr[:,0].numpy()   # train 500만
    y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
    t0=time.time()
    Er=emb.embed(Xr);mu,sd=Er.mean(0),Er.std(0)+1e-6;pca=PCA(50,random_state=0).fit((Er-mu)/sd)
    Z=lambda E:pca.transform((E-mu)/sd)
    Zr=Z(Er);tr=cKDTree(Zr);drr=tr.query(Zr,k=2)[0][:,1]   # real-real LOO NN
    Sr=stats(Xr);np.savez(f'{O}/real_emb.npz',E=Er,Z=Zr,S=Sr,drr=drr)
    P_(f'real emb done {time.time()-t0:.0f}s; real-real NN median {np.median(drr):.3f} p95 {np.percentile(drr,95):.3f}')
    def realism(Xg,Eg,tag):
        Zg=Z(Eg);dg=tr.query(Zg,k=1)[0];Sg=stats(Xg)
        ss=(Sr.mean(0),Sr.std(0)+1e-9)
        r=dict(tag=tag,nn_med=float(np.median(dg)),nn_ratio=float(np.median(dg)/np.median(drr)),
               frac_in_p95=float((dg<=np.percentile(drr,95)).mean()),
               auc_dino=float(auc2(Zr,Zg)),auc_stats=float(auc2((Sr-ss[0])/ss[1],(Sg-ss[0])/ss[1])),
               stat_smd={n:float((Sg[:,j].mean()-Sr[:,j].mean())/Sr[:,j].std()) for j,n in enumerate(SN)},
               real_cover=float(np.median(cKDTree(Zg).query(Zr,k=1)[0])/np.median(drr)))
        P_(f"[{tag}] NN med {r['nn_med']:.3f} (ratio {r['nn_ratio']:.2f}) in-p95 {r['frac_in_p95']:.2f} AUC dino {r['auc_dino']:.3f} stats {r['auc_stats']:.3f} cover {r['real_cover']:.2f}")
        return r,Zg,dg
    RES=[]
    # v1: 넓은 prior 풀
    rng=np.random.default_rng(0);NP=4000
    PP=[sample_prior(rng) for _ in range(NP)];Xp=G(PP,range(10_000,10_000+NP),pool);Ep=emb.embed(Xp)
    P_(f'pool {time.time()-t0:.0f}s')
    r,Zp,_=realism(Xp[:500],Ep[:500],'v1_prior');RES.append(r)
    # v2: 실제 각 장의 최근접 풀 파라미터 -> 새 seed로 재생성
    Zp=Z(Ep);nn=cKDTree(Zp).query(Zr,k=1)[1];P2=[PP[j] for j in nn]
    X2=G(P2,range(20_000,20_500),pool);E2=emb.embed(X2);r,_,_=realism(X2,E2,'v2_nnmatch');RES.append(r)
    # v3: 실제 각 장마다 국소 탐색 (3 라운드 x 10 후보, 고정 seed로 평가) -> 새 seed로 재생성
    W={k:(PRIOR[k][1]-PRIOR[k][0]) for k in KEYS}
    def perturb(p,sc,rr):
        q=dict(p)
        for k in KEYS:
            if PRIOR[k][2]: q[k]=p[k]*np.exp(rr.normal(0,sc*0.6))
            else: q[k]=p[k]+rr.normal(0,sc*W[k]*0.25)
        return clip(q)
    best=list(P2);bd=np.linalg.norm(Z(emb.embed(G(best,range(30_000,30_500),pool)))-Zr,axis=1)
    rr=np.random.default_rng(1)
    for rd,sc in enumerate([0.5,0.3,0.15]):
        for c in range(10):
            cand=[perturb(p,sc,rr) for p in best]
            dc=np.linalg.norm(Z(emb.embed(G(cand,range(30_000,30_500),pool)))-Zr,axis=1)
            imp=dc<bd;bd=np.where(imp,dc,bd);best=[cand[i] if imp[i] else best[i] for i in range(500)]
        P_(f'refine round {rd} mean target dist {bd.mean():.3f} {time.time()-t0:.0f}s')
    X3=G(best,range(40_000,40_500),pool);E3=emb.embed(X3);r,Z3,d3=realism(X3,E3,'v3_refined');RES.append(r)
    # 같은 파라미터, 다른 seed 반복 시 대상 실제 이미지와 거리(파라미터 수준 적합도)
    r['self_target_dist_med']=float(np.median(np.linalg.norm(Z3-Zr,axis=1)))
    json.dump(RES,open(f'{O}/realism.json','w'),indent=1)
    json.dump(best,open(f'{O}/fitted_params.json','w'))
    np.save(f'{O}/fit_target_dist.npy',bd)
    import cv2
    o=np.argsort(y);idx=o[np.linspace(0,499,8).astype(int)]
    im=np.concatenate([np.concatenate([Xr[i] for i in idx],1),np.concatenate([X3[i] for i in idx],1),np.concatenate([Xp[i] for i in range(8)],1)],0)
    cv2.imwrite(f'{O}/real_vs_gen.png',(im*255).astype(np.uint8))
    P_('S1 DONE')
