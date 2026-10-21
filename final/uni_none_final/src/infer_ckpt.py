# Test inference from saved fold ckpts (orig channel, same TTA as vit_arm.py). usage: infer_ckpt.py <run_dir> <ref_bal_d4_dir>
import sys,json,numpy as np,torch,timm,torch.nn as nn,torch.nn.functional as F,cv2
from common import *
D,REF=sys.argv[1],sys.argv[2]; ARCH='vit_base_patch14_dinov2.lvd142m';SIZE=252;dev='mps'
G=np.stack([cv2.imread(f'{R}/test/{i}.png',0) for i in test_ids()]).astype(np.float32)
def apply(g,st):
    C=np.stack([g/255.]*3,1); m=np.array(st['mean'])[None,:,None,None]; s=np.array(st['std'])[None,:,None,None]
    return torch.tensor(((C-m)/s).astype(np.float32))
def d4(x,k):
    x=torch.rot90(x,k%4,(2,3));return torch.flip(x,(3,)) if k>=4 else x
def prep(x): return F.interpolate(x,size=SIZE,mode='bilinear',align_corners=False)
class M(nn.Module):
    def __init__(s):
        super().__init__();s.b=timm.create_model(ARCH,pretrained=False,num_classes=0,img_size=SIZE)
        d=s.b.num_features;s.h=nn.Sequential(nn.Linear(2*d,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,1))
    def forward(s,x):
        t=s.b.forward_features(x);pt=t[:,s.b.num_prefix_tokens:]
        return s.h(torch.cat([t[:,0],pt.mean(1)],1)).squeeze(1)
@torch.no_grad()
def predict(m,X,mu,sd):
    m.eval();P=[]
    for i in range(0,len(X),32):
        xb=X[i:i+32].to(dev);P.append((sum(m(prep(d4(xb,k))) for k in range(8))/8).cpu())
    return torch.cat(P).numpy()*sd+mu
out={};tot=0
for f in range(5):
    ck=torch.load(f'{D}/fold{f}.pt',map_location='cpu',weights_only=False); m=M(); m.load_state_dict(ck['state_dict']); m.to(dev)
    p=predict(m,apply(G,ck['norm']),ck['ymu'],ck['ysd']); tot=tot+p/5
    out[f'fold{f}']=dict(sha256=sha(f'{D}/fold{f}.pt'),maxabs_vs_train_time=float(np.abs(p-np.load(f'{D}/fold{f}.npz')['pt']).max()))
np.save(f'{D}/test_from_ckpt.npy',tot)
rt=np.load(f'{D}/test.npy'); bt=np.load(f'{REF}/test.npy'); bo=np.load(f'{REF}/oof.npy'); no=np.load(f'{D}/oof.npy'); Y=y()
out['test_ckpt_vs_train_time_maxabs']=float(np.abs(tot-rt).max())
out['vs_bal_d4']=dict(test_corr=float(np.corrcoef(tot,bt)[0,1]),test_maxabs=float(np.abs(tot-bt).max()),test_rms=float(np.sqrt(((tot-bt)**2).mean())),
   oof_maxabs=float(np.abs(no-bo).max()),oof_rmse_new=rm(no,Y),oof_rmse_bal_d4=rm(bo,Y),bit_identical_oof=bool(np.array_equal(no,bo)))
json.dump(out,open(f'{D}/ckpt_check.json','w'),indent=1); print(json.dumps(out,indent=1))
