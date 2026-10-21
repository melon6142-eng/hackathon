import os; os.environ['OMP_NUM_THREADS']='2'
import sys;sys.path.insert(0,'lib');from cv import *;from feats import DINO
Y=y()
for n,s in {'std L9-12':[(L,2) for L in range(4)],'mean L9-12':[(L,1) for L in range(4)],'mean+std+gem L9-12':[(L,k) for L in range(4) for k in (1,2,3)],'mean+std L9':[(0,1),(0,2)],'mean+std L10':[(1,1),(1,2)]}.items():
  o,r=nested_ridge(DINO('train',sel=s),Y,alphas=np.logspace(0,6,13),reps=1);print(report(n,o,Y),flush=True)
