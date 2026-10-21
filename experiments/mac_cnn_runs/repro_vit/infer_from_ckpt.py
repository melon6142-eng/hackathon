# rebuild test/OOF predictions from saved ckpts only (no training)
import sys,os,csv,json,numpy as np,torch
tag,chan=sys.argv[1],sys.argv[2]
sys.argv=['x','--arch','vit_base_patch14_dinov2.lvd142m','--size','252','--chan',chan,'--bs','16','--lr','5e-5','--tag','__infer_'+tag]
src=open(os.path.dirname(os.path.abspath(__file__))+'/dino_ft_repro.py').read().split('# fold split fixed')[0]
exec(src)
from sklearn.model_selection import KFold
folds=list(KFold(5,shuffle=True,random_state=42).split(y))
D=f'{R}/cnn/runs/repro_vit/{tag}';oof=np.zeros(len(y));pt=0;rep={}
for f,(tr,va) in enumerate(folds):
    m=M().to(dev);m.load_state_dict(torch.load(f'{D}/fold{f}.pt',map_location=dev))
    pv=predict(m,Xtr[va]);p=predict(m,Xte);oof[va]=pv;pt=pt+p/5
    d=np.load(f'{D}/fold{f}.npz');rep[f]=dict(maxdiff_test=float(np.abs(p-d['pt']).max()),maxdiff_va=float(np.abs(pv-d['pv']).max()))
np.save(f'{D}/oof_ckpt.npy',oof);np.save(f'{D}/test_ckpt.npy',pt)
rep['oof_rmse']=float(np.sqrt(((oof-y)**2).mean()));json.dump(rep,open(f'{D}/ckpt_infer.json','w'),indent=1);print(tag,rep)
import shutil;shutil.rmtree(f'{R}/cnn/runs/repro_vit/__infer_'+tag,ignore_errors=True)
