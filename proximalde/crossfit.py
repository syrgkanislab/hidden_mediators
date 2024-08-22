import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from joblib import Parallel, delayed


def fit_predict_single(X, Y, modelcv, model, cv, semi):
    ''' Runs a single cross-fit prediction.

    Parameters
    ----------
    X : array (nobs, nfeatures)
        The regression features
    Y : array (nobs, noutcomes) or (nobs,)
        The target outcomes
    modelcv : obj
        An estimator object. When `semi=True`, this object will be
        fitted on all the data and then the attribute `alpha_` of
        the fitted object will be used and will be assigned to the
        estimator object model, which will then be used to
        cross_val_predict the outcomes. When `semi=True` the object
        needs to have attribute `alpha_` after fit.
    model : obj or None
        An estimator object. When `semi=True`, this object will be
        used as described in the `modelcv` parameter. When `semi=False`
        this object will be ignored and can be set to None.
    cv : cross-validation fold generator
        Any cross validation option that is acceptable by
        `sklearn.model_selection.check_cv`.
    semi : bool
        Whether semi-cross-fitting or cross-fitting will be performed.

    Returns
    -------
    cvpreds : array same shape as `Y`
        Out-of-fold predictions for each input sample.
    '''
    if semi:
        modelcv = clone(modelcv).fit(X, Y)
        if not hasattr(modelcv, 'alpha_'):
            raise AttributeError("When `semi=True`, the `modelcv` object needs "
                                 "to have attribute `alpha_` after being fitted.")
        alpha = modelcv.alpha_
        model = clone(model)
        if not hasattr(model, 'alpha'):
            raise AttributeError("When `semi=True`, the `model` object needs "
                                 "to have init parameter `alpha`.")
        model.alpha = alpha
    else:
        model = clone(modelcv)
    return cross_val_predict(model, X, Y, cv=cv).reshape(Y.shape)


def fit_predict(X, Y, modelcv, model, cv, semi, multitask, n_jobs, verbose):
    ''' Produce out-of-fold predictions of `Y`. Allows for either multitasking
    or for separate fitting for each target in `Y`, when `Y` contains many
    targets.

    Parameters
    ----------
    X : array (nobs, nfeatures)
        The regression features
    Y : array (nobs, noutcomes) or (nobs,)
        The target outcomes
    modelcv : obj
        An estimator object. When `semi=True`, this object will be
        fitted on all the data and then the attribute `alpha_` of
        the fitted object will be used and will be assigned to the
        estimator object model, which will then be used to
        cross_val_predict the outcomes. When `semi=True` the object
        needs to have attribute `alpha_` after fit.
    model : obj or None
        An estimator object. When `semi=True`, this object will be
        used as described in the `modelcv` parameter. When `semi=False`
        this object will be ignored and can be set to None.
    cv : cross-validation fold generator
        Any cross validation option that is acceptable by
        `sklearn.model_selection.check_cv`.
    semi : bool
        Whether semi-cross-fitting or cross-fitting will be performed.
    multitask : bool
        Whether to fit a single joint model for all targets or run a
        separate cross-fit procedure for each target using clones
        of the `modelcv` and `model` objects for each one.
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
    if multitask or (len(Y.shape) == 1):
        return fit_predict_single(X, Y, modelcv, model, cv, semi)
    else:
        Ypreds = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(fit_predict_single)(X, Y[:, i], modelcv, model, cv, semi)
            for i in range(Y.shape[1]))
        return np.column_stack(Ypreds)
