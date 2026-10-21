# Phase3 쌍 풀: 같은 구조(base) + 다른 촬영 후처리 2개. fold 학습 이미지 파라미터만 사용. 5000 구조 x 2
import sys,json,zlib,numpy as np
sys.path.insert(0,'.')
from multiprocessing import Pool
import gen2,sampler
from sklearn.model_selection import KFold
rs,f=int(sys.argv[1]),int(sys.argv[2]);tag=f'pair_rs{rs}_f{f}';H_=zlib.crc32(tag.encode())
idx=list(KFold(5,shuffle=True,random_state=rs).split(np.arange(500)))[f][0];P5=json.load(open('fitted_params_v5.json'))
ps=sampler.sample(P5,idx,5000,H_);r=np.random.default_rng(H_+1);pp=[dict(P5[int(r.choice(idx))]) for _ in range(5000)]
NK=['illum','blur','noise','tex_a','tex_s','sharp']
def g(a):
    p,q,s=a;b=gen2.base(p,s).astype(np.float32);p2=dict(p);[p2.__setitem__(k,q[k]) for k in NK]
    return np.stack([(gen2.post(b,p,s)*255).round(),(gen2.post(b,p2,s+1)*255).round()]).astype(np.uint8)
if __name__=='__main__':
    base=2*10**9+H_%10**5*10000
    with Pool(10) as pl: X=np.stack(pl.map(g,[(p,pp[i],base+2*i) for i,p in enumerate(ps)],chunksize=16))
    np.savez(f'pool_{tag}.npz',X=X,T=np.stack([sampler.tgt(p) for p in ps]));print('done',X.shape)
