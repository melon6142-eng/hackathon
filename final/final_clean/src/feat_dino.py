# BL1 DINO features. Identical recipe to runs/dino/feat_dino.py (per-image only: /255, fixed ImageNet mean/std, D4 TTA).
# Weights: Meta torch.hub dinov2_vitb14 (Apache-2.0), loaded from local hub cache (offline).
import torch,numpy as np,os,cv2,json
from common import *
torch.set_num_threads(6); torch.manual_seed(0)
H=os.path.expanduser('~/.cache/torch/hub')
m=torch.hub.load(f'{H}/facebookresearch_dinov2_main','dinov2_vitb14',source='local').eval()
ck=f'{H}/checkpoints/dinov2_vitb14_pretrain.pth'
json.dump({'dinov2_vitb14_pretrain.pth':sha(ck),'hubconf.py':sha(f'{H}/facebookresearch_dinov2_main/hubconf.py')},open(f'{O}/weights_dinov2_sha256.json','w'),indent=1)
mean=np.array([0.485,0.456,0.406]);std=np.array([0.229,0.224,0.225]);size=252
def feats(x):
    fs=[]
    for p,c in m.get_intermediate_layers(x,n=4,return_class_token=True):
        fs+=[c,p.mean(1),p.std(1),p.clamp(min=1e-6).pow(3).mean(1).pow(1/3)]
    return torch.cat(fs,1)
for sp,ids in [('train',train_ids()),('test',test_ids())]:
    F=[]
    for k in range(0,len(ids),8):
        ims=[cv2.resize(cv2.imread(f'{R}/{sp}/{i}.png',0),(size,size),interpolation=cv2.INTER_AREA)/255. for i in ids[k:k+8]]
        x=torch.tensor(np.stack([(np.stack([a]*3,-1)-mean)/std for a in ims]).transpose(0,3,1,2),dtype=torch.float32)
        acc=0
        with torch.no_grad():
            for r in range(4):
                for fl in [False,True]:
                    xx=torch.rot90(x,r,(2,3)); xx=torch.flip(xx,(3,)) if fl else xx; acc=acc+feats(xx)
        F.append((acc/8).numpy())
        if k%200==0: print(sp,k,flush=True)
    np.save(f'{O}/feat/dino_{sp}.npy',np.concatenate(F).astype(np.float32))
print('DONE')
