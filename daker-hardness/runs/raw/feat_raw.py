import numpy as np,pandas as pd,cv2,os
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../open')
def f(p):
    im=cv2.imread(p,0).astype(np.float32); d={}
    for s in [0,1,2,4,8]:
        b=cv2.GaussianBlur(im,(0,0),s) if s else im
        d[f'mean{s}']=b.mean(); d[f'std{s}']=b.std()
        for q in [1,5,25,50,75,95,99]: d[f'q{q}_{s}']=np.percentile(b,q)
        gx=cv2.Sobel(b,cv2.CV_32F,1,0);gy=cv2.Sobel(b,cv2.CV_32F,0,1);gm=np.hypot(gx,gy)
        d[f'g{s}']=gm.mean(); d[f'g95_{s}']=np.percentile(gm,95)
    d['noise']=np.std(im-cv2.GaussianBlur(im,(0,0),1))
    return d
for sp in ['train','test']:
    ids=sorted(x[:-4] for x in os.listdir(f'{R}/{sp}'))
    df=pd.DataFrame([f(f'{R}/{sp}/{i}.png') for i in ids]);df.insert(0,'ID',ids)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),f'raw_{sp}.csv'),index=False)
