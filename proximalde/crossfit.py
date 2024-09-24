import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from joblib import Parallel, delayed


def fit_predict_single(X, Y, isbinary, model_regression, model_classification, cv, semi):
    ''' Runs a single cross-fit prediction.

    Parameters
    ----------
    X : array (nobs, nfeatures)
        The regression features
    Y : array (nobs, noutcomes) or (nobs,)
        The target outcomes
    isbinary : bool
        Whether the target is categorical and the model is classifier
    model_regression : obj or None
        An regression estimator object. When `semi=True`, this object will be
        fitted on all the data and then clones of model.best_estimator_
        will be used to produce cross-val-predictions. Hence, when
        `semi=True` the estimator needs to have attribute `best_estimator_`
        after being fitted. Will be used for all continuous targets
    model_classification : obj or None
        An regression estimator object. When `semi=True`, this object will be
        fitted on all the data and then clones of model.best_estimator_
        will be used to produce cross-val-predictions. Hence, when
        `semi=True` the estimator needs to have attribute `best_estimator_`
        after being fitted. Will be used for all binary targets
    cv : cross-validation fold generator
        Any cross validation option that is acceptable by
        `sklearn.model_selection.check_cv`.
    semi : bool
        Whether semi-cross-fitting or cross-fitting will be performed.

    Return:
    -------
    cvpreds : array same shape as `Y`
        Out-of-fold predictions for each input sample.
    '''
    model = model_classification if isbinary else model_regression
    if semi:
        model = clone(model).fit(X, Y)
        if not hasattr(model, 'best_estimator_'):
            raise AttributeError("When `semi=True`, the `model` object needs "
                                 "to have attribute `best_estimator_` after being fitted.")
        model = clone(model.best_estimator_)
    else:
        model = clone(model)

    if isbinary:
        import warnings 
        warnings.filterwarnings("ignore")   
        for x,y in cv:
            if Y[y].mean()==0 and Y[x].mean() != 0:
                Y[x] = 0
            elif Y[y].mean()==1 and Y[x].mean() != 1:
                Y[x] = 1
        # For rare classes, will error if Y[train] contains all 0's (or all 1's) but Y[test] doesn't
        return cross_val_predict(model, X, Y, cv=cv, method='predict_proba')[:, 1].reshape(Y.shape)
    else:
        return cross_val_predict(model, X, Y, cv=cv).reshape(Y.shape)


def fit_predict(X, Y, isbinary, model_regression, model_classification, cv, semi, n_jobs, verbose):
    ''' Produce out-of-fold predictions of `Y`. Allows for either multitasking
    or for separate fitting for each target in `Y`, when `Y` contains many
    targets.

    Parameters
    ----------
    X : array (nobs, nfeatures)
        The regression features
    Y : array (nobs, noutcomes) or (nobs,)
        The target outcomes
    isbinary : ArrayLike[bool]
        For each target Y, whether it is binary
    model_regression : obj or None
        An regression estimator object. When `semi=True`, this object will be
        fitted on all the data and then clones of model.best_estimator_
        will be used to produce cross-val-predictions. Hence, when
        `semi=True` the estimator needs to have attribute `best_estimator_`
        after being fitted. Will be used for all continuous targets
    model_classification : obj or None
        An regression estimator object. When `semi=True`, this object will be
        fitted on all the data and then clones of model.best_estimator_
        will be used to produce cross-val-predictions. Hence, when
        `semi=True` the estimator needs to have attribute `best_estimator_`
        after being fitted. Will be used for all binary targets
    cv : cross-validation fold generator
        Any cross validation option that is acceptable by
        `sklearn.model_selection.check_cv`.
    semi : bool
        Whether semi-cross-fitting or cross-fitting will be performed.
    n_jobs : int or None
        Number of cores for multiprocessing. None means no parallelism.
        Use -1 to use all cores available.
    verbose : int
        Verbosity of parallel for loops.

    Returns
    -------
    cvpreds : array same shape as `Y`
        Out-of-fold predictions for each input sample.
    '''
    if len(Y.squeeze().shape) == 1:
        return fit_predict_single(X, Y.ravel(), isbinary[0], model_regression, model_classification,
                                cv, semi).reshape(Y.shape)
    else:
        Ypreds = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(fit_predict_single)(X, Y[:, i], isbinary[i], model_regression, model_classification,
                                        cv, semi)
            for i in range(Y.shape[1]))
        return np.column_stack(Ypreds)
