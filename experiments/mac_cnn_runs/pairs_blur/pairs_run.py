import os,json,csv,numpy as np,torch,cv2
os.environ['OMP_NUM_THREADS']='4'
from scipy import ndimage as nd
from scipy.stats import wilcoxon,binomtest,spearmanr
from scipy.optimize import nnls
from sklearn.preprocessing import normalize
R='~/AI/daker';C=f'{R}/cnn/runs';O=f'{C}/pairs_blur';os.makedirs(O,exist_ok=True)
rows=list(csv.DictReader(open(f'{R}/open/train.csv')));ids=[r['ID'] for r in rows];y=np.array([float(r['hardness']) for r in rows])
D=json.load(open(f'{C}/robust_pairs/pair_design.json'))
P=[(c['hi_idx'],c['primary']['mid_idx']) for c in D['candidates']]
Xtr,_=torch.load(f'{R}/cnn/imgs.pt',map_location='cpu');X=Xtr[:,0].numpy().astype(np.float64)*255
def coh(img):
    gx=nd.sobel(img,1);gy=nd.sobel(img,0)
    J11=nd.gaussian_filter(gx*gx,4);J22=nd.gaussian_filter(gy*gy,4);J12=nd.gaussian_filter(gx*gy,4)
    l=np.sqrt((J11-J22)**2+4*J12**2);return float(np.mean(l/(J11+J22+1e-9)))
def feats(img):
    b=nd.gaussian_filter(img,2);zi=(b-np.median(b))/(b.std()+1e-9);zg=(b-146.5)/15.3
    f={}
    for nm,z in [('i',zi),('g',zg)]:
        f[f'st1_{nm}']=(z<-1).mean();f[f'st2_{nm}']=((z>=-1)&(z<-.5)).mean();f[f'st3_{nm}']=((z>=-.5)&(z<.5)).mean();f[f'st4_{nm}']=(z>=.5).mean()
    lab,n=nd.label(zi<-1);sz=np.bincount(lab.ravel())[1:] if n else np.array([0])
    f['dark_nblob']=n;f['dark_maxblob']=sz.max()/img.size;f['dark_meanblob']=sz.mean()/img.size
    f['coherence']=coh(img);f['mean']=img.mean();f['std']=img.std()
    f['hf_std']=(img-nd.gaussian_filter(img,1)).std();f['lap_var']=nd.laplace(img).var()
    gy,gx=np.gradient(img);f['sharp']=np.mean(gx**2+gy**2)
    F=np.abs(np.fft.fftshift(np.fft.fft2(img-img.mean())))**2;yy,xx=np.indices(F.shape)-128;r=np.hypot(yy,xx)
    f['lowfreq_frac']=F[r<16].sum()/F.sum();f['midfreq_frac']=F[(r>=16)&(r<64)].sum()/F.sum()
    # grain proxy: bright-cell segmentation
    lb,nb=nd.label(zi>-0.5);s2=np.bincount(lb.ravel())[1:] if nb else np.array([0]);s2=s2[s2>20]
    f['cell_n']=len(s2);f['cell_meansz']=s2.mean() if len(s2) else 0;f['cell_cv']=s2.std()/s2.mean() if len(s2) else 0
    return f
FE=[feats(X[i]) for i in range(500)];names=list(FE[0]);M=np.array([[d[k] for k in names] for d in FE])
# blend OOF (full-fit NNLS on members, optimistic but only for residual description)
ld=lambda t:np.load(f'{C}/{t}/oof.npy');B=np.column_stack([np.load(f'{R}/cnn/oof_blend1.npy'),np.mean([ld(t) for t in ['dinoB_orig','dinoB_orig_s1','dinoB_orig_s2']],0),np.mean([ld(t) for t in ['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2']],0),np.ones(500)])
pb=B@nnls(B,y)[0];M=np.column_stack([M,pb,y-pb]);names+=['blend_pred','blend_resid']
hi=np.array([a for a,b in P]);mi=np.array([b for a,b in P]);dy=y[hi]-y[mi]
Dm=M[hi]-M[mi];res=[]
rng=np.random.default_rng(0);midpool=np.where((y>=210)&(y<=230))[0]
for j,k in enumerate(names):
    d=Dm[:,j];npos=(d>0).sum();nneg=(d<0).sum();agree=max(npos,nneg)/max(npos+nneg,1)
    ps=binomtest(int(max(npos,nneg)),int(npos+nneg),0.5).pvalue if npos+nneg else 1
    pw=wilcoxon(d).pvalue if np.any(d!=0) else 1
    # null: random mid pairing, sign agreement
    nul=[];
    for _ in range(2000):
        rm=rng.choice(midpool,len(hi));dd=M[hi,j]-M[rm,j];nul.append(max((dd>0).mean(),(dd<0).mean()))
    rho=spearmanr(d,dy)[0]
    res.append(dict(metric=k,median_diff=float(np.median(d)),hi_mean=float(M[hi,j].mean()),mid_mean=float(M[mi,j].mean()),sign_agree=float(agree),dir='hi>mid' if npos>nneg else 'hi<mid',p_sign=float(ps),p_wilcoxon=float(pw),p_null_randpair=float(np.mean(np.array(nul)>=agree)),rho_dy=float(rho)))
def bh(p):
    p=np.array(p);n=len(p);o=np.argsort(p);q=np.empty(n);c=1
    for r,i in enumerate(o[::-1]):
        c=min(c,p[i]*n/(n-r));q[i]=c
    return q
def holm(p):
    p=np.array(p);n=len(p);o=np.argsort(p);q=np.empty(n);c=0
    for r,i in enumerate(o): c=max(c,min(1,p[i]*(n-r)));q[i]=c
    return q
qw=bh([r['p_wilcoxon'] for r in res]);qs=bh([r['p_sign'] for r in res]);hw=holm([r['p_wilcoxon'] for r in res])
for r,a,b,c in zip(res,qw,qs,hw): r['q_bh_wilcoxon']=float(a);r['q_bh_sign']=float(b);r['holm_wilcoxon']=float(c)
res.sort(key=lambda r:r['p_wilcoxon'])
# redundancy vs existing signals across 500
ix={k:i for i,k in enumerate(names)}
for r in res:
    v=M[:,ix[r['metric']]];r['rho_vs_dark']=float(spearmanr(v,M[:,ix['st1_i']])[0]);r['rho_vs_coh']=float(spearmanr(v,M[:,ix['coherence']])[0]);r['rho_vs_resid500']=float(spearmanr(v,M[:,ix['blend_resid']])[0])
# reproducibility: alternative embeddings
z=np.load(f'{C}/feat_dinoB/feats.npz');a3=np.load(f'{C}/exp_abc/feats_a.npz')
def prim(E):
    E=normalize(E[:500]);return np.array([midpool[np.argmax(E[midpool]@E[i])] for i in hi])
alt={'dinov2_Lmid':prim(np.concatenate([z['cls_L0'],z['mean_L0']],1)),'dinov3_Llast':prim(np.concatenate([a3['cls_L1'],a3['mean_L1']],1)),'dinov3_Lmid':prim(np.concatenate([a3['cls_L0'],a3['mean_L0']],1))}
rep={}
top3=[r['metric'] for r in res if r['metric'] not in('blend_pred','blend_resid')][:3]
for nm,m2 in alt.items():
    D2=M[hi]-M[m2];r2=[]
    for j,k in enumerate(names):
        d=D2[:,j];r2.append((wilcoxon(d).pvalue if np.any(d!=0) else 1,k))
    r2.sort();t3=[k for p,k in r2 if k not in('blend_pred','blend_resid')][:3]
    rep[nm]=dict(overlap_primary=float((m2==mi).mean()),top3=t3,top3_shared_with_base=len(set(t3)&set(top3)))
json.dump(dict(pairs=[(ids[a],ids[b]) for a,b in P],mid_unique=int(len(set(mi))),results=res,repro=rep,base_top3=top3),open(f'{O}/pairs_stats.json','w'),indent=2)
# mosaic: 18 rows x (hi, mid, hi CLAHE, mid CLAHE)
cl=cv2.createCLAHE(2.5,(8,8));tiles=[]
for (a,b) in P:
    ia=X[a].clip(0,255).astype(np.uint8);ib=X[b].clip(0,255).astype(np.uint8)
    row=[ia,ib,cl.apply(ia),cl.apply(ib)];row=[cv2.copyMakeBorder(cv2.resize(t,(192,192)),22,2,2,2,cv2.BORDER_CONSTANT,value=255) for t in row]
    lab=[f'{ids[a][-3:]} y={y[a]:.0f} p={pb[a]:.0f}',f'{ids[b][-3:]} y={y[b]:.0f} p={pb[b]:.0f}','hi CLAHE','mid CLAHE']
    for t,s in zip(row,lab): cv2.putText(t,s,(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.45,0,1)
    tiles.append(np.hstack(row))
half=9;cv2.imwrite(f'{O}/pairs_mosaic.png',np.hstack([np.vstack(tiles[:half]),np.full((len(tiles[0])*half,10),255,np.uint8),np.vstack(tiles[half:])]))
print('top metrics');[print(r['metric'],r['dir'],round(r['sign_agree'],2),'pW',round(r['p_wilcoxon'],4),'qBH',round(r['q_bh_wilcoxon'],3),'holm',round(r['holm_wilcoxon'],3),'pNull',round(r['p_null_randpair'],3),'rho_dy',round(r['rho_dy'],2),'rDark',round(r['rho_vs_dark'],2),'rCoh',round(r['rho_vs_coh'],2),'rRes',round(r['rho_vs_resid500'],2)) for r in res[:14]]
print('repro',json.dumps(rep),'base_top3',top3,'mid_unique',len(set(mi)))
