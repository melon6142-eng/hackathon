import os; os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'
import sys;sys.path.insert(0,'lib');from cv import *
Y=y();F=np.load('dino/vitb14_252_train.npy',mmap_mode='r');D=768
blk=lambda L,k: np.asarray(F[:,(L*4+k)*D:(L*4+k+1)*D])
sets={'L4 cls+mean':[(3,0),(3,1)],'L4 all':[(3,k) for k in range(4)],'L3 cls+mean':[(2,0),(2,1)],'L1-4 mean+std':[(L,k) for L in range(4) for k in (1,2)]}
for n,s in sets.items():
  X=np.hstack([blk(*p) for p in s]);o,r=nested_ridge(X,Y,alphas=np.logspace(0,6,13),reps=1);np.save(f'oof_dino_{n.replace(" ","_")}.npy',o);print(report(n,o,Y),flush=True)
