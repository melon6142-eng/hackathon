# re-extract BL1 DINO feats (copy of box runs/dino/feat_dino.py logic) for first 20 train + 20 test ids
import torch,numpy as np,os,cv2,sys
torch.manual_seed(0)
m=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14').eval()
R='~/AI/daker/open';size=252
mean=np.array([0.485,0.456,0.406]);std=np.array([0.229,0.224,0.225])
def feats(x):
    fs=[]
    for p,c in m.get_intermediate_layers(x,n=4,return_class_token=True):
        fs+=[c,p.mean(1),p.std(1),p.clamp(min=1e-6).pow(3).mean(1).pow(1/3)]
    return torch.cat(fs,1)
for sp in ['train','test']:
    ids=sorted(x[:-4] for x in os.listdir(f'{R}/{sp}'))[:20];F=[]
    for k in range(0,20,8):
        ims=[cv2.resize(cv2.imread(f'{R}/{sp}/{i}.png',0),(size,size),interpolation=cv2.INTER_AREA)/255. for i in ids[k:k+8]]
        x=torch.tensor(np.stack([(np.stack([a]*3,-1)-mean)/std for a in ims]).transpose(0,3,1,2),dtype=torch.float32)
        acc=0
        with torch.no_grad():
            for r in range(4):
                for fl in [False,True]:
                    xx=torch.rot90(x,r,(2,3));xx=torch.flip(xx,(3,)) if fl else xx;acc=acc+feats(xx)
        F.append((acc/8).numpy())
    np.save(f're_{sp}20.npy',np.concatenate(F).astype(np.float32))
print('DONE')
