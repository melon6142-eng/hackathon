# 규칙 기반 분리: 입계(얇은 선) / 기공(작고 매우 어두운 둥근 구멍) / 2상(어두운 다각형 셀). train 이미지별 독립 처리만.
import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, cv2
from skimage import filters, segmentation, measure, morphology, restoration
P0=dict(gs=1.0, rp=80, h=2.0, tp=2.5, tph=1.5, amin=6, circ=0.75)
def segment(im, P=P0):
    im=im.astype(np.float32)/255
    sig=float(restoration.estimate_sigma(im))
    den=cv2.bilateralFilter(im,9,0.08,5); den=cv2.GaussianBlur(den,(0,0),P['gs']*(1.0+10*sig))
    ridge=filters.sato(den,sigmas=[1,1.5],black_ridges=True)
    bnd=ridge>np.percentile(ridge,P['rp'])
    skel=morphology.skeletonize(bnd)
    # 기공: 셀 평균 분포 기준이 아니라 픽셀 강도의 아래쪽 꼬리 + 두께 + 원형도
    med=np.median(den); mad=1.4826*np.median(np.abs(den-med))+1e-6; z=(den-med)/mad
    cand=measure.label(z<-P['tp'])
    pore=np.zeros_like(bnd)
    dt=cv2.distanceTransform((z<-P['tp']).astype(np.uint8),cv2.DIST_L2,3)
    for r in measure.regionprops(cand):
        if r.area<P['amin']: continue
        circ=4*np.pi*r.area/(r.perimeter**2+1e-6)
        thick=dt[r.coords[:,0],r.coords[:,1]].max()
        zc=z[r.coords[:,0],r.coords[:,1]]
        if circ>=P['circ'] and r.solidity>=0.9 and thick>=2.0 and r.area<=600 and np.percentile(zc,20)<-(P['tp']+0.5): pore[r.coords[:,0],r.coords[:,1]]=True
    pore=cv2.dilate(pore.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    interior=~(bnd|pore); dist=cv2.distanceTransform(interior.astype(np.uint8),cv2.DIST_L2,3)
    mk=measure.label(morphology.h_maxima(dist,P['h']))
    lab=segmentation.watershed(ridge,mk,mask=~pore)
    # 셀 평균: 입계·기공 제외 내부 픽셀
    inner=lab*(interior)
    n=lab.max()
    cnt=np.bincount(inner.ravel(),minlength=n+1)[1:].astype(float)
    sm=np.bincount(inner.ravel(),weights=den.ravel(),minlength=n+1)[1:]
    A=np.bincount(lab.ravel(),minlength=n+1)[1:].astype(float)
    ok=(cnt>=5)&(A>=20)
    Ic=np.where(cnt>0,sm/np.maximum(cnt,1),np.nan)
    # 2상: 셀 평균이 셀 평균 분포(중앙값, robust sd)보다 tph sd 이상 어두운 셀
    v=Ic[ok]; w=A[ok]
    m0=np.median(v); csd=1.4826*np.median(np.abs(v-m0))+1e-6
    th=m0-P['tph']*csd
    hiI=v[v>=th]; loI=v[v<th]
    sep=(np.median(hiI)-loI.mean())/csd if len(loI)>0 else 0
    dark=np.zeros(n,bool); dark[np.where(ok)[0]]=Ic[ok]<th
    ph=np.isin(lab,np.where(dark)[0]+1)
    return dict(den=den,sig=sig,ridge=ridge,bnd=bnd,skel=skel,pore=pore,lab=lab,A=A,Ic=Ic,ok=ok,dark=dark,ph=ph,sep=sep,th=th,csd=csd)
def overlay(im,S):
    o=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR).astype(np.float32)
    def tint(m,c,a=0.45): o[m]=(1-a)*o[m]+a*np.array(c,np.float32)
    tint(S['ph'],(255,120,0))      # 2상: 파랑(BGR)
    tint(S['skel'],(0,255,255),0.8) # 입계: 노랑
    tint(S['pore'],(0,0,255),0.8)   # 기공: 빨강
    return o.clip(0,255).astype(np.uint8)
