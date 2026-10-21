# v5 생성기: gen_cf.gen 구조 + 미세 고주파 질감(입자 내부 상관 노이즈), unsharp, 8bit 양자화
import sys,os,numpy as np,cv2
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'../gen_cf'))
import gen as G0
H=256
def base(p,seed):
    q=dict(p);q['blur']=0.0;q['noise']=0.0;q['illum']=0.0   # 후처리 전 구조 영상
    return G0.gen(q,seed)  # clip 포함(0~1 범위 내라 영향 미미)
def post(im,p,seed):
    r=np.random.default_rng(seed+99991)
    il=G0._smooth_field(r,48); im=im+p['illum']*il
    if p['blur']>0.05: im=cv2.GaussianBlur(im,(0,0),p['blur'])
    if p.get('tex_a',0)>0:   # 미세 상관 질감 (sigma tex_s px)
        t=cv2.GaussianBlur(r.standard_normal((H,H)).astype(np.float32),(0,0),p['tex_s']);t/=t.std()+1e-9
        im=im+p['tex_a']*t
    if p.get('sharp',0)>0:
        im=im+p['sharp']*(im-cv2.GaussianBlur(im,(0,0),1.0))
    im=im+p['noise']*r.standard_normal((H,H))
    im=np.clip(im,0,1)
    return (np.round(im*255)/255).astype(np.float32)
def gen(p,seed): return post(base(p,seed).astype(np.float32),p,seed)
