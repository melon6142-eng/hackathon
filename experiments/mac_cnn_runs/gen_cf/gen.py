# 합성 미세조직 생성기: 이방성 Voronoi + 휨 + 2상 + 기공 + 입계 + 조명/blur/노이즈
import numpy as np, cv2
from scipy.spatial import cKDTree
H=256
KEYS=['n_grains','aspect','theta','warp','ph_frac','ph_c','pore_frac','pore_r','bnd_w','bnd_c','base','grain_sd','illum','blur','noise']
# 넓은 prior (샘플링용): (lo,hi,log?)
PRIOR={'n_grains':(25,500,1),'aspect':(1.0,3.5,1),'theta':(0,np.pi,0),'warp':(0,6,0),'ph_frac':(0,0.45,0),'ph_c':(0.03,0.2,0),
 'pore_frac':(0.0,0.04,0),'pore_r':(2.5,9,1),'bnd_w':(0.6,2.2,0),'bnd_c':(0.0,0.15,0),'base':(0.45,0.8,0),'grain_sd':(0.0,0.05,0),
 'illum':(0,0.08,0),'blur':(0.3,1.6,0),'noise':(0.005,0.12,1)}
def sample_prior(rng):
    p={}
    for k,(lo,hi,lg) in PRIOR.items(): p[k]=float(np.exp(rng.uniform(np.log(lo),np.log(hi))) if lg else rng.uniform(lo,hi))
    return p
def clip(p):
    q=dict(p)
    for k,(lo,hi,lg) in PRIOR.items():
        if k=='theta': q[k]=q[k]%np.pi
        else: q[k]=float(np.clip(q[k],lo,hi))
    return q
def _smooth_field(rng,s):
    f=rng.standard_normal((H//8+2,H//8+2)); f=cv2.resize(f,(H,H),interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(f,(0,0),s)/ (cv2.GaussianBlur(f,(0,0),s).std()+1e-9)
def gen(p,seed):
    """seed가 같으면 씨앗 위치·기공 위치·노이즈가 고정 -> 변수 하나만 바꾼 반사실 이미지가 가능"""
    r=np.random.default_rng(seed)
    PAD=48; L=H+2*PAD
    U=r.uniform(-PAD,H+PAD,(4000,2))            # 고정 씨앗 목록(앞에서부터 n개 사용)
    gperm=r.permutation(4000); gnoise=r.standard_normal(4000); phn=r.standard_normal(4000)
    PU=r.uniform(0,H,(600,2)); PR=r.lognormal(0,0.25,600); PA=r.uniform(1,1.6,600); PT=r.uniform(0,np.pi,600)
    wx=_smooth_field(r,64); wy=_smooth_field(r,64); il=_smooth_field(r,48); nz=r.standard_normal((H,H))
    rc=np.random.default_rng(seed+7777); FC=_smooth_field(rc,10)   # 16-64px 규모 군집장
    FCi=lambda P: FC[np.clip(P[:,1].astype(int),0,H-1),np.clip(P[:,0].astype(int),0,H-1)]
    hg,cph,cpo=p.get('gs_het',0.0),p.get('ph_clust',0.0),p.get('pore_clust',0.0)
    n=int(round(p['n_grains']*(L/H)**2))
    if hg>0:
        uu=rc.uniform(size=4000); key=uu/np.exp(1.2*hg*FCi(U)); U=U[np.argsort(key)]
    S=U[:n]
    yy,xx=np.mgrid[0:H,0:H].astype(np.float64)
    xx=xx+p['warp']*wx; yy=yy+p['warp']*wy
    c,s=np.cos(p['theta']),np.sin(p['theta']); sa=np.sqrt(p['aspect'])
    def T(x,y): u=c*x+s*y; v=-s*x+c*y; return np.stack([u/sa,v*sa],-1)
    tree=cKDTree(T(S[:,0],S[:,1]))
    d,i=tree.query(T(xx,yy).reshape(-1,2),k=2); d=d.reshape(H,H,2); lab=i[:,0].reshape(H,H)
    # 입계까지 거리 근사(메트릭 공간), 화소 스케일 보정
    gap=(d[...,1]**2-d[...,0]**2)/(2*np.linalg.norm(T(*S[i[:,1]].T)-T(*S[i[:,0]].T),axis=-1).reshape(H,H)+1e-9)
    gap=gap*np.sqrt(sa*1/sa+0)  # ~px
    bprof=np.exp(-(gap/p['bnd_w'])**2)
    area=np.bincount(lab.ravel(),minlength=n).astype(float)
    g=p['base']+p['grain_sd']*gnoise[:n]
    # 2상: 고정 순서대로 면적 누적이 ph_frac에 도달할 때까지
    order=[k for k in gperm if k<n]
    if cph>0:
        rk=np.empty(n); rk[np.array(order)]=np.arange(n)/n
        sc=(1-cph)*rk-cph*0.5*(1+np.tanh(FCi(S)))
        order=list(np.argsort(sc))
    cum=np.cumsum(area[order])/area.sum()
    dk=np.array(order)[cum<=p['ph_frac']]
    g[dk]-=p['ph_c']*(1+0.15*phn[dk])
    im=g[lab]-p['bnd_c']*bprof
    # 기공: 고정 목록에서 면적분율 도달까지
    pm=np.zeros((H,H))
    target=p['pore_frac']*H*H; acc=0; k=0
    if cpo>0:
        CU=rc.uniform(0,H,(3000,2)); key=rc.uniform(size=3000)/np.exp(3*cpo*FCi(CU)); PU=np.where(np.arange(600)[:,None]<600,CU[np.argsort(key)][:600],PU)
    while acc<target and k<600:
        rr=p['pore_r']*PR[k]; a,b=rr*np.sqrt(PA[k]),rr/np.sqrt(PA[k])
        cv2.ellipse(pm,(int(PU[k,0]),int(PU[k,1])),(max(1,int(round(a))),max(1,int(round(b)))),float(np.degrees(PT[k])),0,360,1.0,-1)
        acc+=np.pi*a*b; k+=1
    pm=cv2.GaussianBlur(pm,(0,0),0.8)
    im=im*(1-pm)+(p['base']-p.get('pore_d',0.3))*pm
    im=im+p['illum']*il
    if p['blur']>0.05: im=cv2.GaussianBlur(im,(0,0),p['blur'])
    im=im+p['noise']*nz
    return np.clip(im,0,1).astype(np.float32)
