import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from joblib import Parallel, delayed

def fit_predict_single(X, Y, modelcv, model, splits, semi):
    if semi:
        alpha = clone(modelcv).fit(X, Y).alpha_
        model = clone(model)
        model.alpha = alpha
    else:
        model = clone(modelcv)
    return cross_val_predict(model, X, Y, cv=splits).reshape(Y.shape)

def fit_predict(X, Y, modelcv, model, splits, semi, multitask, n_jobs, verbose):
    ''' Generically project Y onto X, i.e. replace Y with E[Y|X]
    '''
    if (Y.shape[1] > 1) and multitask:
        if semi:
            alpha = clone(modelcv).fit(X, Y).alpha_
            model = clone(model)
            model.alpha = alpha
        else:
            model = clone(modelcv)
        return cross_val_predict(model, X, Y, cv=splits).reshape(Y.shape)
    else:
        Ypreds = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(fit_predict_single)(X, Y[:, i], modelcv, model, splits, semi)
            for i in range(Y.shape[1]))
        return np.column_stack(Ypreds)