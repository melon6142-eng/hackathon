import os; os.environ['OMP_NUM_THREADS']='2'
import sys;sys.path.insert(0,'lib');from cv import y,report;from blend import *;from feats import *
Y=y(); SEL=[(L,k) for L in range(4) for k in (1,2)]
Xa=HX('train'); Xb=DINO('train',sel=SEL)
o,r,rs,w=nested_blend([Xa,Xb],Y); np.save('oof_blend1.npy',o)
print(report('blend HX + dinoB252 L9-12 mean+std',o,Y),'reps',rs.round(3),'w',w.round(3),flush=True)
