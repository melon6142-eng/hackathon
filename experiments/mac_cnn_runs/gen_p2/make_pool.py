# usage: make_pool.py TAG [fold_rs fold]  -> pool_TAG.npz (uint8 imgs, targets, params). fold 지정 시 그 fold의 학습 이미지 파라미터만 사용
import sys,json,numpy as np
sys.path.insert(0,'.')
from multiprocessing import Pool
import gen2,sampler
from sklearn.model_selection import KFold
tag=sys.argv[1];P5=json.load(open('fitted_params_v5.json'))
if len(sys.argv)>2:
    rs,f=int(sys.argv[2]),int(sys.argv[3]);idx=list(KFold(5,shuffle=True,random_state=rs).split(np.arange(500)))[f][0]
else: idx=np.arange(500)
N=10000;import zlib;H_=zlib.crc32(tag.encode());base=H_%10**5*100000+10**8
ps=sampler.sample(P5,idx,N,H_)
def g(a): return (gen2.gen(*a)*255).round().astype(np.uint8)
if __name__=='__main__':
    with Pool(11) as pl: X=np.stack(pl.map(g,[(p,base+i) for i,p in enumerate(ps)],chunksize=16))
    T=np.stack([sampler.tgt(p) for p in ps]);np.savez(f'pool_{tag}.npz',X=X,T=T,idx=idx);json.dump(ps,open(f'pool_{tag}_params.json','w'));print('done',X.shape)
