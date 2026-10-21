import os; os.environ['OMP_NUM_THREADS']='2'
import sys;sys.path.insert(0,'lib');from cv import y;from blend import *;from feats import *;import pandas as pd
Y=y(); SEL=[(L,k) for L in range(4) for k in (1,2)]
p,w=blend_fit_predict([HX('train'),DINO('train',sel=SEL)],Y,[HX('test'),DINO('test',sel=SEL)])
ss=pd.read_csv('./open/sample_submission.csv'); ids=sorted(x[:-4] for x in os.listdir('./open/test'))
assert list(ss.ID)==ids or set(ss.ID)==set(ids)
s=pd.DataFrame({'ID':ids,'hardness':p}).set_index('ID').loc[ss.ID].reset_index()
assert len(s)==1000 and s.hardness.notna().all()
os.makedirs('./out',exist_ok=True); s.to_csv('./out/submission_blend1_cv12.95.csv',index=False)
old=pd.read_csv('./open/mlccfx_v1/submission_d0_blend.csv').set_index('ID').loc[ss.ID]
print('w',w,'mean',p.mean(),'std',p.std(),'corr vs 12.52 sub',np.corrcoef(old.iloc[:,0],s.hardness)[0,1])
