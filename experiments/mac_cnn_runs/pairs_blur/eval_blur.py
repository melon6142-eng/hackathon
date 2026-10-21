import os,json,csv,numpy as np,torch,timm,torch.nn as nn,torch.nn.functional as F
from scipy.optimize import nnls
from sklearn.model_selection import KFold
R='~/AI/daker';C=f'{R}/cnn/runs';O=f'{C}/pairs_blur'
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))]);hi=y>230;lo=y<170
rm=lambda p:float(np.sqrt(((p-y)**2).mean()));ld=lambda t:np.load(f'{C}/{t}/oof.npy')
S=[42,1,2];OT={42:'dinoB_orig',1:'dinoB_orig_s1',2:'dinoB_orig_s2'};BT={s:f'dinoB_orig_blur_s{s}' for s in S}
folds=list(KFold(5,shuffle=True,random_state=42).split(y))
out={'solo':[]};fw=0;fn=0
for s in S:
    a,b=ld(OT[s]),ld(BT[s]);fr=[]
    for tr,va in folds:
        ra=float(np.sqrt(((a[va]-y[va])**2).mean()));rb=float(np.sqrt(((b[va]-y[va])**2).mean()));fr.append((ra,rb));fw+=rb<ra;fn+=1
    out['solo'].append(dict(seed=s,orig=rm(a),blur=rm(b),d=rm(b)-rm(a),orig_hi=float((a-y)[hi].mean()),blur_hi=float((b-y)[hi].mean()),orig_lo=float((a-y)[lo].mean()),blur_lo=float((b-y)[lo].mean()),corr=float(np.corrcoef(a,b)[0,1]),sse_win=float(((b-y)**2<(a-y)**2).mean()),fold_rmse=fr))
out['fold_wins']=f'{fw}/{fn}'
BL1=np.load(f'{R}/cnn/oof_blend1.npy');Oa=np.mean([ld(OT[s]) for s in S],0);Ob=np.mean([ld(BT[s]) for s in S],0)
Ta=np.mean([ld(t) for t in ['dinoB_trans','dinoB_trans_s1','dinoB_trans_s2']],0)
out['avg3']=dict(orig=rm(Oa),blur=rm(Ob),corr=float(np.corrcoef(Oa,Ob)[0,1]))
A1=lambda X:np.column_stack([X,np.ones(len(X))])
def ncv(X):
    r=[];pps=[]
    for s in range(10):
        pp=np.zeros(500)
        for tr,va in KFold(5,shuffle=True,random_state=100+s).split(X): pp[va]=A1(X[va])@nnls(A1(X[tr]),y[tr])[0]
        r.append(rm(pp));pps.append(pp)
    return np.array(r),pps
cfg={'base':np.column_stack([BL1,Oa,Ta]),'replace(Oblur)':np.column_stack([BL1,Ob,Ta]),'add(+Oblur)':np.column_stack([BL1,Oa,Ta,Ob]),'Oavg(orig+blur 6)':np.column_stack([BL1,(Oa+Ob)/2,Ta])}
br,bp=ncv(cfg['base']);out['blend']=[]
for k,X in cfg.items():
    r,p=ncv(X)
    out['blend'].append(dict(cfg=k,ncv=float(r.mean()),sd=float(r.std()),d=float(r.mean()-br.mean()),rep_wins=int((r<br).sum()),hi=float(np.mean([(q-y)[hi].mean() for q in p])),lo=float(np.mean([(q-y)[lo].mean() for q in p])),sse_win=float(np.mean([((q-y)**2<(b-y)**2).mean() for q,b in zip(p,bp)])),w=np.round(nnls(A1(X),y)[0],3).tolist()))
json.dump(out,open(f'{O}/blur_eval.json','w'),indent=2);print('blend done',flush=True)
dev='mps';Xtr,_=torch.load(f'{R}/cnn/imgs.pt');g=Xtr[:,0].numpy().astype(np.float64)
Cc=np.stack([g,g,g],1);mu_,sd_=Cc.mean((0,2,3),keepdims=True),Cc.std((0,2,3),keepdims=True);del Cc
def norm(G): C3=np.stack([G,G,G],1);return torch.tensor(((C3-mu_)/sd_).astype(np.float32))
class M(nn.Module):
    def __init__(s):
        super().__init__();s.b=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=False,num_classes=0,img_size=252)
        d=s.b.num_features;s.h=nn.Sequential(nn.Linear(2*d,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,1))
    def forward(s,x):
        t=s.b.forward_features(x);pt=t[:,s.b.num_prefix_tokens:];return s.h(torch.cat([t[:,0],pt.mean(1)],1)).squeeze(1)
from scipy.ndimage import gaussian_filter
def pert(G,b,n):
    G=G.copy()
    if b: G=np.stack([gaussian_filter(i,b) for i in G])
    if n: G=np.clip(G+np.random.default_rng(0).normal(0,n,G.shape),0,1)
    return G
P=[('clean',0,0),('blur_0.5',.5,0),('blur_1.0',1,0),('noise_0.005',0,.005),('noise_0.01',0,.01),('noise_0.02',0,.02)]
mu,sd=y.mean(),y.std()
def d4(x,k):
    x=torch.rot90(x,k%4,(2,3));return torch.flip(x,(3,)) if k>=4 else x
@torch.no_grad()
def pred(m,X):
    P_=[]
    for i in range(0,len(X),32):
        xb=F.interpolate(X[i:i+32].to(dev),size=252,mode='bilinear',align_corners=False);P_.append((sum(m(d4(xb,k)) for k in range(8))/8).cpu())
    return torch.cat(P_).numpy()*sd+mu
sens={}
for s in S:
    preds={k:np.zeros(500) for k,_,_ in P}
    for f,(tr,va) in enumerate(folds):
        m=M();m.load_state_dict(torch.load(f'{C}/{BT[s]}/fold{f}_ck.pt',map_location='cpu'));m=m.to(dev).eval()
        for k,b,n in P: preds[k][va]=pred(m,norm(pert(g[va],b,n)))
    c=preds['clean'];sens[str(s)]={k:dict(mean_abs_d=float(np.abs(p-c).mean()),corr=float(np.corrcoef(p,c)[0,1]),rmse=rm(p),d_rmse=rm(p)-rm(c),hi_abs_d=float(np.abs(p-c)[hi].mean())) for k,p in preds.items()}
    sens[str(s)]['clean_vs_saved_oof_maxdiff']=float(np.abs(c-ld(BT[s])).max());print('sens',s,flush=True)
out['sens']=sens
json.dump(out,open(f'{O}/blur_eval.json','w'),indent=2);print(json.dumps(out,indent=1))
