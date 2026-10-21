# v4: 각 적합 파라미터의 밝기 평균·대비를 대상 실제 이미지에 맞춤 (진폭 스케일 + base 이동), 2회 반복
import sys,os,json,numpy as np,torch
O=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,O)
from s1_fit import *
AMP=['ph_c','bnd_c','grain_sd','illum','noise']
if __name__=='__main__':
    import emb;from scipy.spatial import cKDTree
    pool=Pool(10);R='~/AI/daker';Xr,_=torch.load(f'{R}/cnn/imgs.pt');Xr=Xr[:,0].numpy()
    best=json.load(open(f'{O}/fitted_params.json'))
    for p in best: p.setdefault('pore_d',0.3)
    for it in range(3):
        Xg=G(best,range(30_000,30_500),pool)
        for i,p in enumerate(best):
            k=np.clip(Xr[i].std()/(Xg[i].std()+1e-6),0.3,3)
            for a in AMP: p[a]=float(p[a]*k)
            p['pore_d']=float(p['pore_d']*k)
            p['base']=float(p['base']+(Xr[i].mean()-Xg[i].mean()))
        print('iter',it,'mean|dm|',np.abs(Xr.mean((1,2))-Xg.mean((1,2))).mean(),'mean|ds|',np.abs(Xr.std((1,2))-Xg.std((1,2))).mean(),flush=True)
    RE=np.load(f'{O}/real_emb.npz');Er=RE['E'];Zr0=RE['Z'];drr=RE['drr'];Sr=RE['S']
    mu,sd=Er.mean(0),Er.std(0)+1e-6;pca=PCA(50,random_state=0).fit((Er-mu)/sd);Z=lambda E:pca.transform((E-mu)/sd);Zr=Z(Er);tr=cKDTree(Zr)
    X4=G(best,range(40_000,40_500),pool);E4=emb.embed(X4);Zg=Z(E4);dg=tr.query(Zg,k=1)[0];Sg=stats(X4);ss=(Sr.mean(0),Sr.std(0)+1e-9)
    r=dict(tag='v4_moment',nn_med=float(np.median(dg)),nn_ratio=float(np.median(dg)/np.median(drr)),frac_in_p95=float((dg<=np.percentile(drr,95)).mean()),
       auc_dino=float(auc2(Zr,Zg)),auc_stats=float(auc2((Sr-ss[0])/ss[1],(Sg-ss[0])/ss[1])),
       stat_smd={n:float((Sg[:,j].mean()-Sr[:,j].mean())/Sr[:,j].std()) for j,n in enumerate(SN)},
       real_cover=float(np.median(cKDTree(Zg).query(Zr,k=1)[0])/np.median(drr)),self_target_dist_med=float(np.median(np.linalg.norm(Zg-Zr,axis=1))))
    print(r,flush=True)
    RES=json.load(open(f'{O}/realism.json'));RES=[x for x in RES if x['tag']!='v4_moment']+[r];json.dump(RES,open(f'{O}/realism.json','w'),indent=1)
    json.dump(best,open(f'{O}/fitted_params_v4.json','w'))
    import csv,cv2
    y=np.array([float(q['hardness']) for q in csv.DictReader(open(f'{R}/open/train.csv'))]);o=np.argsort(y);idx=o[np.linspace(0,499,8).astype(int)]
    X3=G(json.load(open(f'{O}/fitted_params.json')),[40_000+i for i in idx],pool)
    im=np.concatenate([np.concatenate([Xr[i] for i in idx],1),np.concatenate(list(X3),1),np.concatenate([X4[i] for i in idx],1)],0)
    cv2.imwrite(f'{O}/real_vs_gen.png',(np.clip(im,0,1)*255).astype(np.uint8));print('DONE')
