# Retrain ViT-B orig (same recipe as dinoB_orig_s*) saving weights, then eval held-out fold on filtered variants
import sys,os,time,json,numpy as np,torch,timm,torch.nn as nn,torch.nn.functional as F,csv
sys.path.insert(0,os.path.dirname(__file__)); from filt import VARIANTS
from sklearn.model_selection import KFold
seed=int(sys.argv[1]);R='~/AI/daker';O=f'{R}/cnn/runs/freq_ablation/vit_s{seed}';os.makedirs(O,exist_ok=True);dev='mps'
SIZE=252;BS=16;LR=5e-5;EP=30;UNF=4;ARCH='vit_base_patch14_dinov2.lvd142m'
torch.manual_seed(seed);np.random.seed(seed)
y=np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/open/train.csv'))])
Xraw,_=torch.load(f'{R}/cnn/imgs.pt');Xraw=Xraw[:,:1].float()
C0=Xraw.numpy();MU,SD=C0.mean(),C0.std()   # orig channel normalization (train stats, same as build())
def build(X): return ((X.repeat(1,3,1,1)-MU)/SD).float()
mu,sd=y.mean(),y.std();zt=torch.tensor((y-mu)/sd,dtype=torch.float32)
def d4(x,k): x=torch.rot90(x,k%4,(2,3));return torch.flip(x,(3,)) if k>=4 else x
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
        t=s.b.forward_features(x);pt=t[:,s.b.num_prefix_tokens:];return s.h(torch.cat([t[:,0],pt.mean(1)],1)).squeeze(1)
def aug(x):
    x=torch.cat([d4(x[i:i+1],int(np.random.randint(8))) for i in range(len(x))]);return x+(torch.rand(len(x),1,1,1,device=x.device)-.5)*0.2
@torch.no_grad()
def predict(m,X):
    m.eval();P=[]
    for i in range(0,len(X),32):
        xb=X[i:i+32].to(dev);P.append((sum(m(prep(d4(xb,k))) for k in range(8))/8).cpu())
    return torch.cat(P).numpy()*sd+mu
Xtr=build(Xraw)
VX={n:build(fn(Xraw)) for n,fn in VARIANTS}
folds=list(KFold(5,shuffle=True,random_state=42).split(y))
log=open(f'{O}/log.txt','a')
def L(s): print(s,flush=True);log.write(s+'\n');log.flush()
for f,(tr,va) in enumerate(folds):
    res=f'{O}/fold{f}_preds.npz'
    if os.path.exists(res): continue
    t0=time.time();torch.manual_seed(seed+f*1000);np.random.seed(seed+f*1000)
    ck=f'{O}/fold{f}_ck.pt';m=M().to(dev)
    if os.path.exists(ck): m.load_state_dict(torch.load(ck,map_location=dev))
    else:
        bp=[q for q in m.b.parameters() if q.requires_grad]
        opt=torch.optim.AdamW([{'params':bp,'lr':LR},{'params':m.h.parameters(),'lr':LR*10}],weight_decay=0.05)
        spe=int(np.ceil(len(tr)/BS));sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[LR,LR*10],total_steps=spe*EP,pct_start=0.1)
        for ep in range(EP):
            m.train();idx=np.random.permutation(tr)
            for b in range(spe):
                bi=idx[b*BS:(b+1)*BS]
                if len(bi)<2: continue
                out=m(prep(aug(Xtr[bi].to(dev))));l=F.huber_loss(out,zt[bi].to(dev))
                opt.zero_grad();l.backward();opt.step();sch.step()
        torch.save(m.state_dict(),ck)
    P={n:predict(m,VX[n][va]) for n in VX}
    np.savez(res,va=va,**P)
    L(f'fold{f} clean {np.sqrt(((P["clean"]-y[va])**2).mean()):.3f} {time.time()-t0:.0f}s')
L('DONE')
