# Phase2 비교 3종 + 고/저경도 분포 분석 (train 500만; 진단)
import sys,os,json,csv,numpy as np,torch,timm,torch.nn.functional as F
sys.path.insert(0,'.')
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold,cross_val_predict
from sklearn.metrics import average_precision_score,roc_auc_score
from scipy.stats import ks_2samp,spearmanr
import sampler
R='~/AI/daker';y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))]);hi=y>230;lo=y<170
X=np.load('train_imgs.npy');O=[]
def P(s): print(s,flush=True);O.append(s)
# (1) 구조 분포: 실제 per-image 적합 파라미터 vs 합성 10k
P5=json.load(open('fitted_params_v5.json'));pa=json.load(open('pool_all_params.json'))
Tr=np.stack([sampler.tgt(dict(p,gs_het=p.get('gs_het',0),ph_clust=p.get('ph_clust',0))) for p in P5]);Tg=np.stack([sampler.tgt(p) for p in pa])
P('## (1) 구조 파라미터 분포: 실제 500(이미지별 적합값) vs 합성 10k\n| 파라미터 | 실제 중앙값[p5,p95] | 합성 중앙값[p5,p95] | KS |\n|---|---|---|---|')
for j,k in enumerate(sampler.TGT):
    a,b=Tr[:,j],Tg[:,j];P(f'| {k} | {np.median(a):.3f} [{np.percentile(a,5):.3f},{np.percentile(a,95):.3f}] | {np.median(b):.3f} [{np.percentile(b,5):.3f},{np.percentile(b,95):.3f}] | {ks_2samp(a,b).statistic:.2f} |')
P('(gs_het/ph_clust는 실제에 적합되지 않은 변수라 합성에서 U(0,0.8)로 뿌림 → 비교 불가)')
# (2) 특징 비교
dev='mps';MU,SD=float(X.mean()),float(X.std())
def feats(sd=None):
    m=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=True,num_classes=0,img_size=252)
    if sd: m.load_state_dict(torch.load(sd),strict=False)
    m=m.eval().to(dev);out=[]
    with torch.no_grad():
        for i in range(0,500,32):
            x=torch.tensor((X[i:i+32]-MU)/SD)[:,None].repeat(1,3,1,1);x=F.interpolate(x,size=252,mode='bilinear',align_corners=False).to(dev)
            t=m.forward_features(x);out.append(torch.cat([t[:,0],t[:,1:].mean(1)],1).cpu().numpy())
    return np.concatenate(out)
fn='feat_cmp.npz'
if os.path.exists(fn): d=np.load(fn);Fd,Fs=d['Fd'],d['Fs']
else: Fd=feats();Fs=feats('pre_rs42_f0.pt');np.savez(fn,Fd=Fd,Fs=Fs)
def z(A): return (A-A.mean(0))/(A.std(0)+1e-6)
def cka(A,B):
    A=A-A.mean(0);B=B-B.mean(0);return np.linalg.norm(A.T@B)**2/(np.linalg.norm(A.T@A)*np.linalg.norm(B.T@B))
kf=KFold(5,shuffle=True,random_state=0)
def r2pred(A,B):
    Bp=cross_val_predict(RidgeCV(alphas=np.logspace(0,5,11)),z(A),z(B),cv=kf);return float(1-((Bp-z(B))**2).sum()/(z(B)**2).sum())
P(f'\n## (2) DINOv2 원본 vs 합성 파라미터 사전학습(pre_rs42_f0) 특징, 실제 500장\n- linear CKA: {cka(z(Fd),z(Fs)):.3f}\n- DINO→새특징 선형 예측 R²(5fold ridge): {r2pred(Fd,Fs):.3f}\n- 새특징→DINO 선형 예측 R²: {r2pred(Fs,Fd):.3f}')
bl=np.load('~/AI/daker/cnn/oof_blend1.npy') if False else None
base=np.mean([np.load(f'../{d}/oof.npy') for d in ('dinoB_orig','dinoB_orig_s1','dinoB_orig_s2')],0);res=y-base
for nm,Fm in (('DINO원본',Fd),('새특징',Fs),('새특징 중 DINO로 설명 안되는 부분',Fs-cross_val_predict(RidgeCV(alphas=np.logspace(0,5,11)),z(Fd),Fs,cv=kf))):
    rp=cross_val_predict(RidgeCV(alphas=np.logspace(0,6,13)),z(Fm),res,cv=kf);hp=cross_val_predict(RidgeCV(alphas=np.logspace(0,6,13)),z(Fm),y,cv=kf)
    P(f'- {nm}: 경도 ridge CV RMSE {np.sqrt(((hp-y)**2).mean()):.2f}, ViT orig 3seed 잔차 설명 R² {1-((rp-res)**2).mean()/res.var():+.3f}')
# 고/저경도 위치 (파라미터 공간, 특징 공간)
P('\n## 고경도 18 / 저경도 38 분포 (n이 작아 참고용)\n| 파라미터 | 고경도 중앙값 | 중간 중앙값 | 저경도 중앙값 | 고vs중간 AUC |\n|---|---|---|---|---|')
mid=~hi&~lo
for j,k in enumerate(sampler.TGT[:6]+['tex_a']):
    jj=sampler.TGT.index(k);a=Tr[:,jj]
    P(f'| {k} | {np.median(a[hi]):.3f} | {np.median(a[mid]):.3f} | {np.median(a[lo]):.3f} | {roc_auc_score(np.r_[np.ones(hi.sum()),np.zeros(mid.sum())],np.r_[a[hi],a[mid]]):.2f} |')
from sklearn.decomposition import PCA
for nm,Fm in (('DINO',Fd),('새특징',Fs),('파라미터',Tr[:,[0,1,2,3,4,5,8]])):
    Zp=PCA(min(10,Fm.shape[1]),random_state=0).fit_transform(z(Fm));Zh=Zp[hi];D=np.linalg.norm(Zh[:,None]-Zh[None],axis=-1);Dm=np.linalg.norm(Zh[:,None]-Zp[mid][None],axis=-1)
    nnh=np.argsort(np.linalg.norm(Zp[:,None]-Zp[None],axis=-1),1)[:,1:6]
    P(f'- {nm} PCA10: 고경도끼리 평균거리 {D[np.triu_indices(18,1)].mean():.2f} vs 고-중간 {Dm.mean():.2f}; 고경도 이미지의 5-NN 중 고경도 비율 {hi[nnh[hi]].mean():.2f} (무작위 {hi.mean():.2f})')
from sklearn.cluster import KMeans
Zh=PCA(5,random_state=0).fit_transform(z(Fd))[hi];
for k in (2,3):
    from sklearn.metrics import silhouette_score;lab=KMeans(k,n_init=10,random_state=0).fit_predict(Zh);P(f'- 고경도 18장 DINO PCA5 KMeans k={k}: silhouette {silhouette_score(Zh,lab):.2f}, 크기 {np.bincount(lab).tolist()}')
open('p2_analysis.md','w').write('\n'.join(O)+'\n')
