import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
def nested(est, grid, X, Y, reps=3):
    oofs=[]
    for r in range(reps):
        oof=np.zeros(len(Y))
        for tr,va in KFold(5,shuffle=True,random_state=42+r).split(X):
            gs=GridSearchCV(make_pipeline(StandardScaler(),est),grid,cv=KFold(5,shuffle=True,random_state=0),scoring='neg_mean_squared_error',n_jobs=2)
            mu=Y[tr].mean(); gs.fit(X[tr],Y[tr]-mu); oof[va]=gs.predict(X[va])+mu
        oofs.append(oof)
    return np.array(oofs).mean(0), np.mean([np.sqrt(((o-Y)**2).mean()) for o in oofs])
