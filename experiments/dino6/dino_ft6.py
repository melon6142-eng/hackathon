# dino_ft6: based on dino_ft5. --chan {orig,trans,mask,maskcoh}; --aux W (multitask: log grain area, aspect ratio, dark frac)
# also saves rotation-TTA preds (8-way D4 x extra angles via grid_sample reflection) as pv_rot/pt_rot
import argparse,os,time,math,numpy as np,torch,timm,torch.nn as nn,torch.nn.functional as F,csv,cv2
from sklearn.model_selection import KFold
p=argparse.ArgumentParser()
p.add_argument('--arch',default='vit_base_patch14_dinov2.lvd142m');p.add_argument('--size',type=int,default=252)
p.add_argument('--chan',default='mask');p.add_argument('--unfreeze',type=int,default=4)
p.add_argument('--epochs',type=int,default=30);p.add_argument('--bs',type=int,default=16)
p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--aux',type=float,default=0.0)
p.add_argument('--angles',default='22.5,45,67.5')
p.add_argument('--tag',required=True);p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
R='~/AI/daker';O=f'{R}/cnn/runs/{a.tag}';os.makedirs(O,exist_ok=True);dev='mps'
torch.manual_seed(a.seed);np.random.seed(a.seed)
rows=list(csv.DictReader(open(f'{R}/open/train.csv')));y=np.array([float(r['hardness']) for r in rows])
Xtr,Xte=torch.load(f'{R}/cnn/imgs.pt')
from scipy.ndimage import distance_transform_edt as edt
def dmask(i):
    b=cv2.GaussianBlur(i,(0,0),2);zz=(b-np.median(b))/(b.std()+1e-6)
    return np.where(zz<-1.0,1.0,np.where(zz<-0.5,0.5,0.0))
def coh(i):
    b=cv2.GaussianBlur(i,(0,0),1.0);gx=cv2.Sobel(b,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(b,cv2.CV_32F,0,1,ksize=3)
    s=4.0;jxx=cv2.GaussianBlur(gx*gx,(0,0),s);jyy=cv2.GaussianBlur(gy*gy,(0,0),s);jxy=cv2.GaussianBlur(gx*gy,(0,0),s)
    return np.sqrt((jxx-jyy)**2+4*jxy**2)/(jxx+jyy+1e-6)
def auxfeat(m):
    dk=(m>0).astype(np.uint8);n,lab,st,_=cv2.connectedComponentsWithStats(dk,8)
    ars=[];areas=[]
    for k in range(1,n):
        A=st[k,cv2.CC_STAT_AREA]
        if A<10: continue
        ys,xs=np.nonzero(lab==k);c=np.cov(np.stack([xs,ys]).astype(float));ev=np.linalg.eigvalsh(c)
        ars.append(np.sqrt(max(ev[1],1e-6)/max(ev[0],1e-6)));areas.append(A)
    if not areas: return [0.,1.,dk.mean()]
    w=np.array(areas,float);return [np.log(np.average(w,weights=w)),np.average(ars,weights=w),dk.mean()]
def build(X,lo,hi):
    g=(X[:,0].numpy()*255);o=g/255.;AX=[]
    if a.chan=='orig': C=np.stack([o,o,o],1)
    elif a.chan in('mask','maskcoh'):
        Ms=[];Ts=[]
        for i in g.astype(np.float32):
            m=dmask(i);Ms.append(m);AX.append(auxfeat(m))
            if a.chan=='mask': dk=m>0;Ts.append(np.clip((edt(dk)-edt(~dk))/10.,-1,1))
            else: Ts.append(coh(i))
        C=np.stack([o,np.array(Ms),np.array(Ts)],1)
    else:
        cl=np.stack([cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8)).apply(np.clip(i,0,255).astype(np.uint8)) for i in g])/255.
        cs=np.power(np.clip((g-lo)/(hi-lo),0,1),0.85);C=np.stack([o,cl,cs],1)
    if not AX: AX=[auxfeat(dmask(i)) for i in g.astype(np.float32)] if a.aux>0 else [[0,0,0]]*len(g)
    return torch.tensor(((C-C.mean((0,2,3),keepdims=True))/C.std((0,2,3),keepdims=True)).astype(np.float32)),np.array(AX,float)
pix=(Xtr.numpy().ravel()*255)[np.random.default_rng(0).choice(Xtr.numel(),2_000_000)];lo,hi=np.percentile(pix,[1,99])
Xtr,AXtr=build(Xtr,lo,hi);Xte,_=build(Xte,lo,hi)
AXz=torch.tensor((AXtr-AXtr.mean(0))/(AXtr.std(0)+1e-6),dtype=torch.float32)
mu,sd=y.mean(),y.std();z=(y-mu)/sd
def d4(x,k):
    x=torch.rot90(x,k%4,(2,3));return torch.flip(x,(3,)) if k>=4 else x
def rot(x,deg):
    t=math.radians(deg);th=torch.tensor([[math.cos(t),-math.sin(t),0],[math.sin(t),math.cos(t),0]],dtype=x.dtype,device=x.device).expand(len(x),2,3)
    return F.grid_sample(x,F.affine_grid(th,x.shape,align_corners=False),padding_mode='reflection',align_corners=False)
def prep(x): return F.interpolate(x,size=a.size,mode='bilinear',align_corners=False)
class M(nn.Module):
    def __init__(s):
        super().__init__();s.b=timm.create_model(a.arch,pretrained=True,num_classes=0,img_size=a.size)
        for q in s.b.parameters(): q.requires_grad=False
        for blk in s.b.blocks[-a.unfreeze:]:
            for q in blk.parameters(): q.requires_grad=True
        for q in s.b.norm.parameters(): q.requires_grad=True
        d=s.b.num_features;s.h=nn.Sequential(nn.Linear(2*d,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,1))
        s.ah=nn.Linear(2*d,3)
    def forward(s,x,both=False):
        t=s.b.forward_features(x);f=torch.cat([t[:,0],t[:,s.b.num_prefix_tokens:].mean(1)],1)
        o=s.h(f).squeeze(1);return (o,s.ah(f)) if both else o
def aug(x):
    x=torch.cat([d4(x[i:i+1],int(np.random.randint(8))) for i in range(len(x))])
    b=(torch.rand(len(x),1,1,1,device=x.device)-.5)*0.2;return x+b
ANG=[float(v) for v in a.angles.split(',') if v]
@torch.no_grad()
def predict(m,X):
    m.eval();P=[];PR=[]
    for i in range(0,len(X),16):
        xb=X[i:i+16].to(dev);base=sum(m(prep(d4(xb,k))) for k in range(8))
        ex=sum(m(prep(d4(rot(xb,ang),k))) for ang in ANG for k in range(8)) if ANG else 0
        P.append((base/8).cpu());PR.append(((base+ex)/(8*(1+len(ANG)))).cpu())
    return torch.cat(P).numpy()*sd+mu,torch.cat(PR).numpy()*sd+mu
folds=list(KFold(5,shuffle=True,random_state=42).split(y))
log=open(f'{O}/log.txt','a')
def L(s): print(s,flush=True);log.write(s+'\n');log.flush()
L(f'== start {time.ctime()} {vars(a)} lo={lo:.1f} hi={hi:.1f}')
zt=torch.tensor(z,dtype=torch.float32)
for f,(tr,va) in enumerate(folds):
    res=f'{O}/fold{f}.npz'
    if os.path.exists(res): L(f'fold{f} done, skip');continue
    t0=time.time();torch.manual_seed(a.seed+f*1000);np.random.seed(a.seed+f*1000);m=M().to(dev)
    bp=[q for q in m.b.parameters() if q.requires_grad];hp=list(m.h.parameters())+list(m.ah.parameters())
    opt=torch.optim.AdamW([{'params':bp,'lr':a.lr},{'params':hp,'lr':a.lr*10}],weight_decay=0.05)
    spe=int(np.ceil(len(tr)/a.bs));sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[a.lr,a.lr*10],total_steps=spe*a.epochs,pct_start=0.1)
    for ep in range(a.epochs):
        m.train();idx=np.random.permutation(tr);tl=0
        for b in range(spe):
            bi=idx[b*a.bs:(b+1)*a.bs]
            if len(bi)<2: continue
            xb=aug(Xtr[bi].to(dev));o,ao=m(prep(xb),both=True);l=F.huber_loss(o,zt[bi].to(dev))
            if a.aux>0: l=l+a.aux*F.huber_loss(ao,AXz[bi].to(dev))  # aux targets are rotation-invariant scalars
            opt.zero_grad();l.backward();opt.step();sch.step();tl+=l.item()
        if ep%10==9: L(f'fold{f} ep{ep} loss {tl/spe:.4f} t={time.time()-t0:.0f}')
    pv,pvr=predict(m,Xtr[va]);pt,ptr=predict(m,Xte);np.savez(res,va=va,pv=pv,pt=pt,pv_rot=pvr,pt_rot=ptr)
    L(f'FOLD{f} RMSE {np.sqrt(((pv-y[va])**2).mean()):.3f} rotTTA {np.sqrt(((pvr-y[va])**2).mean()):.3f} {time.time()-t0:.0f}s')
oof=np.zeros(len(y));oofr=np.zeros(len(y));pt=0;ptr=0
for f in range(5): d=np.load(f'{O}/fold{f}.npz');oof[d['va']]=d['pv'];oofr[d['va']]=d['pv_rot'];pt=pt+d['pt']/5;ptr=ptr+d['pt_rot']/5
np.save(f'{O}/oof.npy',oof);np.save(f'{O}/test.npy',pt);np.save(f'{O}/oof_rot.npy',oofr);np.save(f'{O}/test_rot.npy',ptr);hi_=y>230
for nm,q in(('d4',oof),('rot',oofr)): L(f'ALL OOF[{nm}] RMSE {np.sqrt(((q-y)**2).mean()):.3f} hi_bias {(q-y)[hi_].mean():.1f} lo_bias {(q-y)[y<170].mean():.1f}')
