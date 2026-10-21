# ViT-B DINOv2 partial fine-tune. Based on cnn/runs/repro_vit/dino_ft_repro.py (same model/optim/aug/TTA/hyperparams).
# LEAKAGE FIXES: (1) per-channel normalization mean/std and (2) trans-channel p1/p99 window are fitted on the
# fold-TRAIN images only and applied unchanged to that fold's validation images and to the test images
# (old code: each set normalized by its own statistics -> test-derived stats; p1/p99 from all 500 train incl. val fold).
import hashlib,argparse,os,time,json,numpy as np,torch,timm,torch.nn as nn,torch.nn.functional as F,cv2
from sklearn.model_selection import KFold
from common import *
p=argparse.ArgumentParser()
p.add_argument('--chan',default='orig');p.add_argument('--seed',type=int,default=42);p.add_argument('--fs',type=int,default=42)
p.add_argument('--tag',required=True);p.add_argument('--nockpt',action='store_true')
a=p.parse_args(); ARCH='vit_base_patch14_dinov2.lvd142m';SIZE=252;UNF=4;EP=30;BS=16;LR=5e-5
OD=f'{O}/vit/{a.tag}';os.makedirs(OD,exist_ok=True);dev='mps'
torch.manual_seed(a.seed);np.random.seed(a.seed)
yv=y()
def load(sp,ids): return np.stack([cv2.imread(f'{R}/{sp}/{i}.png',0) for i in ids]).astype(np.float32)  # raw 0..255
Gtr,Gte=load('train',train_ids()),load('test',test_ids())
CL=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8))
def channels(g,lo,hi):
    o=g/255.
    if a.chan=='orig': return np.stack([o,o,o],1)
    cl=np.stack([CL.apply(np.clip(i,0,255).astype(np.uint8)) for i in g])/255.   # per-image op
    cs=np.power(np.clip((g-lo)/(hi-lo),0,1),0.85); return np.stack([o,cl,cs],1)
def fit_stats(g):
    pix=g.ravel()[np.random.default_rng(0).choice(g.size,min(2_000_000,g.size),replace=False)];lo,hi=np.percentile(pix,[1,99])
    C=channels(g,lo,hi); return dict(lo=float(lo),hi=float(hi),mean=C.mean((0,2,3)).tolist(),std=C.std((0,2,3)).tolist())
def apply(g,st):
    C=channels(g,st['lo'],st['hi']); m=np.array(st['mean'])[None,:,None,None]; s=np.array(st['std'])[None,:,None,None]
    return torch.tensor(((C-m)/s).astype(np.float32))
mu,sd=yv.mean(),yv.std()   # label scaling; replaced per fold below with fold-train stats
def d4(x,k):
    x=torch.rot90(x,k%4,(2,3));return torch.flip(x,(3,)) if k>=4 else x
def prep(x): return F.interpolate(x,size=SIZE,mode='bilinear',align_corners=False)
class M(nn.Module):
    def __init__(s):
        super().__init__();s.b=timm.create_model(ARCH,pretrained=True,num_classes=0,img_size=SIZE)
        for q in s.b.parameters(): q.requires_grad=False
        for blk in s.b.blocks[-UNF:]:
            for q in blk.parameters(): q.requires_grad=True
        for q in s.b.norm.parameters(): q.requires_grad=True
        d=s.b.num_features;s.h=nn.Sequential(nn.Linear(2*d,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,1))
    def forward(s,x):
        t=s.b.forward_features(x);pt=t[:,s.b.num_prefix_tokens:]
        return s.h(torch.cat([t[:,0],pt.mean(1)],1)).squeeze(1)
def aug(x):
    x=torch.cat([d4(x[i:i+1],int(np.random.randint(8))) for i in range(len(x))])
    b=(torch.rand(len(x),1,1,1,device=x.device)-.5)*0.2;return x+b
@torch.no_grad()
def predict(m,X,mu,sd):
    m.eval();P=[]
    for i in range(0,len(X),32):
        xb=X[i:i+32].to(dev);P.append((sum(m(prep(d4(xb,k))) for k in range(8))/8).cpu())
    return torch.cat(P).numpy()*sd+mu
folds=list(KFold(5,shuffle=True,random_state=a.fs).split(yv))
log=open(f'{OD}/log.txt','a')
def L(s): print(s,flush=True);log.write(s+'\n');log.flush()
L(f'== start {time.ctime()} {vars(a)}')
for f,(tr,va) in enumerate(folds):
    res=f'{OD}/fold{f}.npz'
    if os.path.exists(res): L(f'fold{f} done, skip');continue
    t0=time.time();torch.manual_seed(a.seed+f*1000);np.random.seed(a.seed+f*1000)
    st=fit_stats(Gtr[tr]); json.dump(st,open(f'{OD}/fold{f}_norm.json','w'))
    Xtr=apply(Gtr,st)   # rows tr used for training; rows va only predicted
    Xte=apply(Gte,st)
    fmu,fsd=yv[tr].mean(),yv[tr].std(); z=(yv-fmu)/fsd; zt=torch.tensor(z,dtype=torch.float32)
    m=M().to(dev); bp=[q for q in m.b.parameters() if q.requires_grad]
    opt=torch.optim.AdamW([{'params':bp,'lr':LR},{'params':m.h.parameters(),'lr':LR*10}],weight_decay=0.05)
    spe=int(np.ceil(len(tr)/BS));sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[LR,LR*10],total_steps=spe*EP,pct_start=0.1)
    for ep in range(EP):
        m.train();idx=np.random.permutation(tr);tl=0
        for b in range(spe):
            bi=idx[b*BS:(b+1)*BS]
            if len(bi)<2: continue
            out=m(prep(aug(Xtr[bi].to(dev))));l=F.huber_loss(out,zt[bi].to(dev))
            opt.zero_grad();l.backward();opt.step();sch.step();tl+=l.item()
        if ep%10==9: L(f'fold{f} ep{ep} loss {tl/spe:.4f} t={time.time()-t0:.0f}')
    pv=predict(m,Xtr[va],fmu,fsd);pt=predict(m,Xte,fmu,fsd)
    if not a.nockpt:
        ck=f'{OD}/fold{f}.pt';torch.save({'state_dict':m.state_dict(),'norm':st,'ymu':fmu,'ysd':fsd},ck);L(f'CKPT fold{f} sha256 '+sha(ck))
    np.savez(res,va=va,pv=pv,pt=pt); L(f'FOLD{f} RMSE {rm(pv,yv[va]):.3f} {time.time()-t0:.0f}s')
oof=np.zeros(len(yv));pt=0
for f in range(5): d=np.load(f'{OD}/fold{f}.npz');oof[d['va']]=d['pv'];pt=pt+d['pt']/5
np.save(f'{OD}/oof.npy',oof);np.save(f'{OD}/test.npy',pt)
L(f'ALL OOF RMSE {rm(oof,yv):.3f} hi_bias {hb(oof,yv):.1f} lo_bias {lb(oof,yv):.1f}')
