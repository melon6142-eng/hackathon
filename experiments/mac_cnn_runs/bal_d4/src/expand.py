# decide fold-seed expansion: arms whose 3-seed mean beats seed-matched baseline by >=0.05
import json;R=json.load(open('~/AI/daker/cnn/runs/bal_d4/results.json'))
print(' '.join(a for a,r in R.items() if a!='base(uni_d4b)' and len(r['seeds'])==3 and r['base_matched']-r['rmse']>=0.05))
