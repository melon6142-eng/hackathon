import sys,os,numpy as np,pandas as pd
sys.path.insert(0,'~/AI/daker/phase_sep')
from multiprocessing import Pool
def hcf(im):
    from feat import feats; from seg import P0
    return feats((np.clip(im,0,1)*255).astype(np.uint8),P0)
if __name__=='__main__':
    X=np.load(sys.argv[1]);
    with Pool(10) as p: R=p.map(hcf,list(X),chunksize=4)
    pd.DataFrame(R).fillna(0).to_csv(sys.argv[2],index=False);print('hc done',len(R))
