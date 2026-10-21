# Nested CV eval for dino7 heads vs base BL1+orig3+trans3
import numpy as np, pandas as pd, os
from scipy.optimize import nnls
from sklearn.model_selection import KFold

D='./runs'
H=f'{D}/overnight'
y=pd.read_csv('./open/train.csv')['hardness'].values
sub=pd.read_csv('./out/submission_blend1_cv12.95.csv')
ld=lambda t,k: np.load(f'{D}/{t}/{k}.npy')
av=lambda ts,k: np.mean([ld(t,k) for t in ts],0)
r=lambda p,t: np.sqrt(((p-t)**2).mean())
hi=y>230; lo=y<170

O3=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','dino_ft2/dinoB_orig_s2']
T3=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','dino_ft2/dinoB_trans_s2']
O4=O3+['overnight/dinoB_orig_s3']
T4=T3+['overnight/dinoB_trans_s3']

# also load new heads
new={
 'dinoB_orig_s3': (np.load(f'{H}/dinoB_orig_s3/oof.npy'), np.load(f'{H}/dinoB_orig_s3/test.npy')),
 'dinoB_trans_s3': (np.load(f'{H}/dinoB_trans_s3/oof.npy'), np.load(f'{H}/dinoB_trans_s3/test.npy')),
 'd7_cls_trans': (np.load(f'{H}/d7_cls_trans/oof.npy'), np.load(f'{H}/d7_cls_trans/test.npy')),
 'd7_mean_trans': (np.load(f'{H}/d7_mean_trans/oof.npy'), np.load(f'{H}/d7_mean_trans/test.npy')),
}

B=np.column_stack([np.load(f'{D}/oof_blend1.npy'), av(O3,'oof'), av(T3,'oof')])
BT=np.column_stack([sub.iloc[:,1].values, av(O3,'test'), av(T3,'test')])
names_base=['BL1','Borig3','Btrans3']

def ncv(X, R=10):
    A=np.column_stack([X, np.ones(len(y))])
    out=[]
    for s in range(R):
        pp=np.zeros(len(y))
        for tr,va in KFold(5, shuffle=True, random_state=100+s).split(A):
            pp[va]=A[va] @ nnls(A[tr], y[tr])[0]
        out.append((r(pp,y), (pp-y)[hi].mean(), (pp-y)[lo].mean()))
    return np.array(out).mean(0)

def full_w(X):
    A=np.column_stack([X, np.ones(len(y))])
    return nnls(A,y)[0]

def maxcorr(X, col):
    # max abs corr of last col with previous cols
    c=np.corrcoef(X.T)
    return float(np.max(np.abs(c[-1,:-1]))) if X.shape[1]>1 else 0.0

base=ncv(B)
print(f'BASE nCV={base[0]:.3f} hi={base[1]:.1f} lo={base[2]:.1f}')
print(f'BASE full_w={full_w(B)}')

rows=[]
def report(label, X, oof_solo=None, XT=None, names=None):
    m=ncv(X)
    w=full_w(X)
    solo=r(oof_solo,y) if oof_solo is not None else float('nan')
    mc=maxcorr(X, -1) if X.shape[1]>3 else (maxcorr(X,-1) if X.shape[1]>1 else 0)
    # for multi-col expansions, report max pairwise among new vs base
    if X.shape[1]>3:
        c=np.corrcoef(X.T)
        # max corr of any new col (from idx 3) with any base (0..2)
        if X.shape[1]>3:
            mc=float(np.max(np.abs(c[3:,:3]))) if X.shape[1]>3 else mc
    delta=m[0]-base[0]
    ws=' '.join(f'{wi:.3f}' for wi in w)
    print(f'| {label} | {solo:.3f} | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} | mc={mc:.3f} | w=[{ws}] |')
    rows.append((label, solo, m[0], delta, m[1], m[2], mc, w, X, XT, names))
    return m

L=['| 구성 | 단독 OOF | nCV(10×5) | Δ | hi | lo | maxcorr | NNLS w |',
   '|---|---|---|---|---|---|---|---|',
   f'| base(BL1+Borig3+Btrans3) | - | {base[0]:.3f} | 0 | {base[1]:.1f} | {base[2]:.1f} | - | {full_w(B)} |']

# A) add each new head alone
print('=== A: +each head ===')
for k,(o,t) in new.items():
    m=report(f'+{k}', np.column_stack([B,o]), o, np.column_stack([BT,t]), names_base+[k])
    L.append(f'| +{k} | {r(o,y):.3f} | {m[0]:.3f} | {m[0]-base[0]:+.3f} | {m[1]:.1f} | {m[2]:.1f} | {maxcorr(np.column_stack([B,o]),-1):.3f} | {full_w(np.column_stack([B,o]))} |')

# B) expand seed averages
print('=== B: expand seeds ===')
Bo4=av(O4,'oof'); Bt4=av(T4,'oof')
Bo4t=av(O4,'test'); Bt4t=av(T4,'test')
Bo3=av(O3,'oof'); Bt3=av(T3,'oof')
configs_B={
 'orig4+trans3': (np.column_stack([np.load(f'{D}/oof_blend1.npy'), Bo4, Bt3]),
                  np.column_stack([sub.iloc[:,1].values, Bo4t, av(T3,'test')])),
 'orig3+trans4': (np.column_stack([np.load(f'{D}/oof_blend1.npy'), Bo3, Bt4]),
                  np.column_stack([sub.iloc[:,1].values, av(O3,'test'), Bt4t])),
 'orig4+trans4': (np.column_stack([np.load(f'{D}/oof_blend1.npy'), Bo4, Bt4]),
                  np.column_stack([sub.iloc[:,1].values, Bo4t, Bt4t])),
}
for lab,(X,XT) in configs_B.items():
    # solo = N/A for blends of avgs; report avgs solo as mean of components? skip solo
    m=ncv(X); w=full_w(X); delta=m[0]-base[0]
    print(f'| {lab} | - | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} | w={w} |')
    L.append(f'| expand {lab} | - | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} | - | {w} |')
    rows.append((f'expand_{lab}', float('nan'), m[0], delta, m[1], m[2], 0., w, X, XT, names_base))

# C) replace / add pool heads
print('=== C: pool replace/add ===')
# replace trans avg with d7_cls / d7_mean
for k in ['d7_cls_trans','d7_mean_trans']:
    o,t=new[k]
    X=np.column_stack([np.load(f'{D}/oof_blend1.npy'), Bo3, o])
    XT=np.column_stack([sub.iloc[:,1].values, av(O3,'test'), t])
    m=ncv(X); w=full_w(X); delta=m[0]-base[0]; mc=maxcorr(X,-1)
    print(f'| replace_trans_with_{k} | {r(o,y):.3f} | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} | mc={mc:.3f} | w={w} |')
    L.append(f'| replace Bt←{k} | {r(o,y):.3f} | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} | {mc:.3f} | {w} |')
    rows.append((f'repl_{k}', r(o,y), m[0], delta, m[1], m[2], mc, w, X, XT, ['BL1','Borig3',k]))
    # add in addition
    X2=np.column_stack([B,o]); XT2=np.column_stack([BT,t])
    m2=ncv(X2); w2=full_w(X2); d2=m2[0]-base[0]; mc2=maxcorr(X2,-1)
    print(f'| +{k}_on_base | {r(o,y):.3f} | {m2[0]:.3f} | {d2:+.3f} | {m2[1]:.1f} | {m2[2]:.1f} | mc={mc2:.3f} | w={w2} |')
    # already in A for + alone

# D) promising combos
print('=== D: combos ===')
# expand both + cls + mean as extra cols
X=np.column_stack([np.load(f'{D}/oof_blend1.npy'), Bo4, Bt4, new['d7_cls_trans'][0], new['d7_mean_trans'][0]])
XT=np.column_stack([sub.iloc[:,1].values, Bo4t, Bt4t, new['d7_cls_trans'][1], new['d7_mean_trans'][1]])
m=ncv(X); w=full_w(X); delta=m[0]-base[0]
print(f'| orig4+trans4+cls+mean | - | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} | w={w} |')
L.append(f'| orig4+trans4+cls+mean | - | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} | - | {w} |')
rows.append(('o4t4clsmean', float('nan'), m[0], delta, m[1], m[2], 0., w, X, XT, None))

# drop seed2 from base then add s3 (from earlier LOO insight)
# O without s2: orig={42,s1} + s3; T without s2: + s3
O_ds2=['dino_ft/dinoB_orig','dino_ft2/dinoB_orig_s1','overnight/dinoB_orig_s3']
T_ds2=['dino_ft/dinoB_trans','dino_ft2/dinoB_trans_s1','overnight/dinoB_trans_s3']
X=np.column_stack([np.load(f'{D}/oof_blend1.npy'), av(O_ds2,'oof'), av(T_ds2,'oof')])
XT=np.column_stack([sub.iloc[:,1].values, av(O_ds2,'test'), av(T_ds2,'test')])
m=ncv(X); w=full_w(X); delta=m[0]-base[0]
print(f'| drop_s2_add_s3 | - | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} | w={w} |')
L.append(f'| drop_s2 + s3 avg | - | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} | - | {w} |')
rows.append(('drop_s2_add_s3', float('nan'), m[0], delta, m[1], m[2], 0., w, X, XT, names_base))

# LOO with s3 in 4-seed: drop each
print('=== LOO among 4 seeds ===')
for drop in range(4):
    Oo=[O4[i] for i in range(4) if i!=drop]
    To=[T4[i] for i in range(4) if i!=drop]
    X=np.column_stack([np.load(f'{D}/oof_blend1.npy'), av(Oo,'oof'), av(To,'oof')])
    m=ncv(X); delta=m[0]-base[0]
    print(f'| LOO drop seed-idx {drop} | - | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} |')
    L.append(f'| LOO4 drop idx{drop} | - | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} | - | - |')

# NNLS over all individual seeds + cls + mean
all_tags=O4+T4+['overnight/d7_cls_trans','overnight/d7_mean_trans']
X=np.column_stack([np.load(f'{D}/oof_blend1.npy')]+[ld(t,'oof') for t in all_tags])
XT=np.column_stack([sub.iloc[:,1].values]+[ld(t,'test') for t in all_tags])
m=ncv(X); w=full_w(X); delta=m[0]-base[0]
print(f'| BL1+8seeds+cls+mean cols | - | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} |')
print('  w=', w)
L.append(f'| BL1+8seeds+cls+mean | - | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} | - | {w} |')
rows.append(('all_cols', float('nan'), m[0], delta, m[1], m[2], 0., w, X, XT, None))

# orig4+trans4 + cls only
X=np.column_stack([np.load(f'{D}/oof_blend1.npy'), Bo4, Bt4, new['d7_cls_trans'][0]])
XT=np.column_stack([sub.iloc[:,1].values, Bo4t, Bt4t, new['d7_cls_trans'][1]])
m=ncv(X); w=full_w(X); delta=m[0]-base[0]
print(f'| orig4+trans4+cls | - | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | w={w} |')
L.append(f'| orig4+trans4+cls | - | {m[0]:.3f} | {delta:+.3f} | {m[1]:.1f} | {m[2]:.1f} | - | {w} |')
rows.append(('o4t4cls', float('nan'), m[0], delta, m[1], m[2], 0., w, X, XT, None))

# solo OOF of new heads
print('=== solo OOF / hi bias ===')
for k,(o,t) in new.items():
    print(f'  {k}: OOF={r(o,y):.3f} hi={(o-y)[hi].mean():.1f} lo={(o-y)[lo].mean():.1f}')

# correlations of new with base
print('=== corr new vs base ===')
for k,(o,_) in new.items():
    print(f'  {k}: BL1={np.corrcoef(B[:,0],o)[0,1]:.3f} Bo={np.corrcoef(B[:,1],o)[0,1]:.3f} Bt={np.corrcoef(B[:,2],o)[0,1]:.3f}')

# find best
best=min(rows, key=lambda z: z[2])
print(f'\nBEST config: {best[0]} nCV={best[2]:.3f} Δ={best[3]:+.3f}')

open(f'{H}/ev_d7.md','w').write('\n'.join(L)+'\n')
print('wrote', f'{H}/ev_d7.md')

# candidate if <=12.55
if best[2] <= 12.55:
    X,XT=best[8],best[9]
    A=np.column_stack([X, np.ones(len(y))])
    w=nnls(A,y)[0]
    pt=np.column_stack([XT, np.ones(len(XT))]) @ w
    s=sub.copy(); s.iloc[:,1]=pt
    fn=f'{H}/cand_{best[0]}_cv{best[2]:.2f}.csv'
    s.to_csv(fn, index=False)
    print('CAND', fn, w)
else:
    print('no candidate (best nCV > 12.55)')
