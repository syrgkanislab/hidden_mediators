import numpy as np
import warnings
import scipy.linalg
from sklearn.base import BaseEstimator, clone


def _check_input(*args):
    ''' Ensures that all matrices are 2d and returns
    2d variant of any column vector as a (n, 1) array
    '''
    # check that all variables have the same samples
    if len(np.unique([arg.shape[0] for arg in args if arg is not None])) > 1:
        raise AttributeError('All input variables need to have the same number '
                             'of samples')
    # reshape all variables to be 2d matrices
    return (arg.reshape(-1, 1) if (arg is not None and len(arg.shape) == 1) else arg for arg in args)


def covariance(X, Z):
    ''' Calculate covariance matrix of two random vectors.
    Each row of X, Z contains a sample of the two vectors.
    '''
    assert X.shape[0] == Z.shape[0], "First dimension must match!"
    return (X - X.mean(axis=0, keepdims=True)).T @ (Z - Z.mean(axis=0, keepdims=True)) / X.shape[0]


def svd_critical_value(Z, X, *, alpha=0.05, mc_samples=1000):
    ''' Calculates critical value on the singular values of a covariance
    matrix of a pair of random vectors, above which they can be deemed
    as non-zero statistically.

    Parameters
    ----------
    Z : ArrayLike[n, pz]
    X : ArrayLike[n, px]
    alpha : float, optional (default=0.05)
        confidence level
    mc_samples : int, optional (default=1000)
        monte carlo samples use to estimate critical value

    Returns
    -------
    critical_value : float
        The critical value for a statistically non-zero singular value of
        the covariance Cov(Z, X)
    '''
    # the line below creates a matrix of (n, px * pz), which could be quite large
    # we need this matrix to calculate the "covariance" of the entries of the covariance
    # matrix En[X Z'], whose eigenvalues we then want to compute.
    n, pz = Z.shape
    px = X.shape[1]

    # if data too large, we do a monte carlo approximation of the eigenvalues
    # though in the end we divide these eigenvalues appropriately by the original
    # size n
    if n * px * pz > 1e9:
        subset = np.random.choice(n, size=int(1e9 // (px * pz)), replace=False)
        Z, X = Z[subset], X[subset]
        warnings.warn("Due to large sample size and proxy dimension, we performed "
                      "monte-carlo approximation of critical value using "
                      f"random subset of n={len(subset)} samples and then re-scaled "
                      "appropriately to account for the larger original sample size.")

    cZX = (Z.reshape(Z.shape + (1,)) * X.reshape(X.shape + (1,)).transpose((0, 2, 1))).reshape(Z.shape[0], -1)
    cZX = cZX - cZX.mean(axis=0, keepdims=True)
    cZXcov = cZX.T @ cZX / cZX.shape[0]

    # here we compute eigenvalues of a symmetric (px * pz, px * pz) matrix
    # this can also be a bit slow. We scale by the original size n, not the
    # potentially subsampled size.
    eigs = scipy.linalg.eigvalsh(cZXcov / n)

    # calculate the critical value via monte carlo simulation of the percentile
    # of the weighted sum of independent chi-square distributed variables
    samples = np.random.chisquare(df=1, size=(mc_samples, len(eigs))) @ eigs
    critical = np.sqrt(np.percentile(samples, (1 - alpha) * 100))
    return critical


class CVWrapper(BaseEstimator):

    def __init__(self, *, modelcv, model, params):
        ''' All params are assumed to be scalar and their values
        will be forced to scalars.
        '''
        self.modelcv = modelcv
        self.model = model
        self.params = params

    def fit(self, X, y):
        # a fitted object on the passed data
        self.modelcv_ = clone(self.modelcv).fit(X, y)

        # an unfitted object with the best hyperparams
        self.best_estimator_ = clone(self.model)
        for key in self.params:
            value = getattr(self.modelcv_, f'{key}_')
            if hasattr(value, '__len__'):
                value = value[0]
            setattr(self.best_estimator_, key, value)

        if hasattr(self.modelcv_, 'classes_'):
            self.classes_ = self.modelcv_.classes_

        return self

    def predict(self, X):
        return self.modelcv_.predict(X)

    def predict_proba(self, X):
        return self.modelcv_.predict_proba(X)
