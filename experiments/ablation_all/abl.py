import os;os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,sys,warnings,json;warnings.filterwarnings('ignore')
sys.path.insert(0,'./runs');os.chdir('./runs')
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel,delayed
from feats import hx,DINO
O='./open';y=pd.read_csv(f'{O}/train.csv').hardness.values;n=500
def tr500(f):
    d=pd.read_csv(f);d=d[d.ID.str.startswith('TRAIN')].sort_values('ID');return d.drop(columns='ID').select_dtypes('number')
h,t,g,r=hx('train')
from sklearn.preprocessing import PolynomialFeatures
top=pd.concat([h,g,r],axis=1);c=top.corrwith(pd.Series(y)).abs().sort_values().tail(14).index
poly=PolynomialFeatures(2,include_bias=False).fit_transform(((top[c]-top[c].mean())/top[c].std()).values)
gen=tr500('gen/gen_feats.csv');NZ=[k for k in gen.columns if 'noise' in k or k.startswith('sig')]
F={'base(밝기+결정립)':pd.concat([h.drop(columns='sharp'),g],axis=1).values,
 '질감(tex)':t,'노이즈·블러':np.column_stack([gen[NZ].values,h[['sharp']].values]),
 '생성규칙지표(gen,노이즈제외)':gen.drop(columns=NZ).values,'경계(bd)':tr500('gen/bd_feats.csv').values,
 '형태(morph)':tr500('gen/morph_feats.csv').values,'자기상관(acf)':tr500('gen/acf_feats.csv').values,
 '대비변환(ctr)':tr500('ablate_new/ctr_feats.csv').values,'raw통계':r.values,'poly2(상위14)':poly,
 '어두운마스크(dm)':tr500('dark_mask/dm_feats.csv').values,'밝기단계비율':pd.read_csv('stage_ablate/ratio_feats.csv').iloc[:500].select_dtypes('number').values,
 '방향·덩어리(feat3)':tr500('joaegi/feat3.csv').values,
 'DINO임베딩(frozen,BL1과 동일층)':DINO('train',sel=[(L,k) for L in range(4) for k in (1,2)]),
 'ConvNeXt-S임베딩(frozen)':np.load(f'{O}/combo_v1/feat/convnext_s_train.npy')}
for k in F:
    X=np.asarray(F[k],float);X=np.where(np.isfinite(X),X,np.nan);med=np.nanmedian(X,0);X=np.where(np.isnan(X),np.nan_to_num(med),X);F[k]=X[:,X.std(0)>0]
ld=lambda p:np.load(p)
av=lambda ts:np.mean([ld(f'dino_ft{x}/oof.npy') for x in ts],0)
M={'BL1':ld('oof_blend1.npy'),'ViT-B 원본(3seed평균)':av(['/dinoB_orig','2/dinoB_orig_s1','2/dinoB_orig_s2']),
 'ViT-B 변환(3seed평균)':av(['/dinoB_trans','2/dinoB_trans_s1','2/dinoB_trans_s2']),
 'ViT-B 변환 u6':av(['2/dinoB_trans_u6']),'ViT-B 변환 u8':av(['2/dinoB_trans_u8']),
 'ViT-S 원본':av(['/dino_orig']),'ViT-S 변환':av(['/dino_trans']),'ViT-S 변환(가중)':av(['/dino_trans_w']),
 'ConvNeXt 미세조정':av(['/cnx_full_w']),'EffNet-B0 미세조정':av(['/effb0_full']),'LDS(d4_lds)':ld('dino_ft3/d4_lds/oof.npy'),
 'gen LGB':ld('gen/oof_lgb_2.npy'),'gen Ridge':ld('gen/oof_ridge_2.npy')}
z=np.load('label_ssl/oof_ssl.npz')
for k in z: M[f'SSL {k}']=z[k]
for k in ['isotonic','qmap','qreg','stretch']: M[f'보정 {k}']=ld(f'cal/oof_{k}.npy')
for k in M: assert M[k].shape==(n,),k
rm=lambda p:np.sqrt(((p-y)**2).mean())
def ridge(): return make_pipeline(StandardScaler(),RidgeCV(np.logspace(-1,6,15)))
def rfit(X,tr,va):
    io=np.zeros(len(tr))
    for a,b in KFold(5,shuffle=True,random_state=1).split(tr): io[b]=ridge().fit(X[tr[a]],y[tr[a]]).predict(X[tr[b]])
    return io,ridge().fit(X[tr],y[tr]).predict(X[va])
def nn(Atr,Ava,tr):
    w=nnls(np.column_stack([Atr,np.ones(len(Atr))]),y[tr])[0];return np.column_stack([Ava,np.ones(len(Ava))])@w
def rankmatch(ptr,ytr,p,a):
    q=np.clip(np.searchsorted(np.sort(ptr),p)/len(ptr),.5/len(ptr),1-.5/len(ptr));return (1-a)*p+a*np.quantile(ytr,q)
BK=['BL1','ViT-B 원본(3seed평균)','ViT-B 변환(3seed평균)']
HXG={'hand':h.values,'tex':t,'grain':g.values,'poly2':poly}
def one(s):
    P={};fo=list(KFold(5,shuffle=True,random_state=100+s).split(np.zeros(n)))
    def put(k,va,p): P.setdefault(k,np.zeros(n))[va]=p
    for tr,va in fo:
        bio,bva=rfit(F['base(밝기+결정립)'],tr,va);put('A|base',va,bva)
        for k,X in F.items():
            if k.startswith('base'): continue
            _,p=rfit(np.hstack([F['base(밝기+결정립)'],X]),tr,va);put('A|+'+k,va,p)
            _,p=rfit(X,tr,va);put('C|'+k,va,p)
        for k,o in M.items(): put('A|+'+k,va,nn(np.column_stack([bio,o[tr]]),np.column_stack([bva,o[va]]),tr))
        # rank match on base (alpha by inner CV)
        AL=np.linspace(0,1,11);e=np.zeros(11)
        for a,b in KFold(5,shuffle=True,random_state=7).split(tr):
            ia,ib=tr[a],tr[b];e+=[((rankmatch(bio[a],y[ia],bio[b],al)-y[ib])**2).sum() for al in AL]
        put('A|+랭크매칭',va,rankmatch(bio,y[tr],bva,AL[e.argmin()]))
        # B: best blend minus components
        C=np.column_stack([M[k] for k in BK]);pb=nn(C[tr],C[va],tr);put('B|최고블렌드',va,pb)
        for i,k in enumerate(BK): j=[x for x in range(3) if x!=i];put('B|-'+k,va,nn(C[tr][:,j],C[va][:,j],tr))
        e=np.zeros(11);ptr_in=np.zeros(len(tr))
        for a,b in KFold(5,shuffle=True,random_state=7).split(tr): ptr_in[b]=nn(C[tr[a]],C[tr[b]],tr[a])
        for a,b in KFold(5,shuffle=True,random_state=8).split(tr): e+=[((rankmatch(ptr_in[a],y[tr[a]],ptr_in[b],al)-y[tr[b]])**2).sum() for al in AL]
        put('B|+랭크매칭',va,rankmatch(ptr_in,y[tr],pb,AL[e.argmin()]))
        # BL1 rebuilt (HX ridge + DINO ridge) and drop each HX group
        dio,dva=rfit(F['DINO임베딩(frozen,BL1과 동일층)'],tr,va);Bo,Bt=M[BK[1]],M[BK[2]]
        def bl(hio,hva,usedino=True):
            A=[hio]+([dio] if usedino else [])+[Bo[tr],Bt[tr]];V=[hva]+([dva] if usedino else [])+[Bo[va],Bt[va]]
            return nn(np.column_stack(A),np.column_stack(V),tr)
        HX=np.hstack(list(HXG.values()));hio,hva=rfit(HX,tr,va);put('B|BL1재구성+B원본+B변환',va,bl(hio,hva))
        put('B|BL1재구성 −DINO임베딩',va,bl(hio,hva,False))
        for k in HXG:
            a,b=rfit(np.hstack([v for kk,v in HXG.items() if kk!=k]),tr,va);put('B|BL1재구성 −'+k,va,bl(a,b))
    return P
R=Parallel(n_jobs=int(os.cpu_count()))(delayed(one)(s) for s in range(10))
import pickle;pickle.dump(R,open('ablation_all/raw.pkl','wb'))
