import os,csv,numpy as np,hashlib
R='~/AI/daker/open'          # read-only competition data
O='~/AI/daker/cnn/runs/bal_d4'
def train_ids(): return [r['ID'] for r in csv.DictReader(open(f'{R}/train.csv'))]
def y(): return np.array([float(r['hardness']) for r in csv.DictReader(open(f'{R}/train.csv'))])
def test_ids(): return [r['ID'] for r in csv.DictReader(open(f'{R}/sample_submission.csv'))]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def rm(p,yv): return float(np.sqrt(((p-yv)**2).mean()))
def hb(p,yv): return float((p-yv)[yv>230].mean())
def lb(p,yv): return float((p-yv)[yv<170].mean())
