import numpy as np
import warnings
import scipy.linalg
from sklearn.base import BaseEstimator, clone, ClassifierMixin, RegressorMixin
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


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


def existence_test_statistic(Z, X, Y, ivreg, random_state=None, reg_exponent=0.2):
    ''' Calculates a test statistic for the existence of a solution
    to an ill-posed linear IV problem.
        E[Z (Y - X'theta)] = 0
    Assumes that ivreg is an ell2 regularized adversarial IV estimator
    of the minimum norm solution to the linear IV problem. The returned
    statistic should follow a chi2(pz) distribution under the null that
    a solution exists.

    Parameters
    ----------
    Z : ArrayLike[n, pz]
    X : ArrayLike[n, px]
    Y : ArrayLike[n,]
    ivreg : instance of adversarial IV estimator
    random_state : None or a random seed, optional (default=None)
    reg_exponent : float, optional (default=None)
        The exponent in the regularization of the eignevalues, i.e.
        we use lambda_j / (lambda_j + n^{-exponent}) when calculating
        the estimate of the projector in the range of the covariance
        SigmaTilde.

    Returns
    -------
    test_statistic : float
        The test statistic. Follows chi2(pz) under the null that a
        solution theta exists.
    '''
    nobs = X.shape[0]
    train, test = train_test_split(np.arange(nobs), test_size=.3, shuffle=True, random_state=random_state)
    ntest = len(test)
    ntrain = len(train)
    ivreg_train = clone(ivreg).fit(Z[train], X[train], Y[train])

    # Estimate of projection matrix SigmaTilde
    # using a regularized SVD decomposition
    Sigma = (Z[train].T @ X[train]) / ntrain
    CovZ = (Z[train].T @ Z[train]) / ntrain
    CovZsqrt = scipy.linalg.sqrtm(CovZ)
    invCovZsqrt = np.linalg.pinv(CovZsqrt)
    SigmaTilde = invCovZsqrt @ Sigma
    U, S, _ = scipy.linalg.svd(SigmaTilde, full_matrices=False)
    P = CovZsqrt @ U @ np.diag(S / (S + 1 / ntrain**(reg_exponent))) @ U.T @ invCovZsqrt

    phi = Z * (Y - X @ ivreg_train.coef_.reshape(-1, 1))
    phi[train] = phi[train] @ P.T
    moments = np.mean(phi[test], axis=0)
    phi[test] = phi[test] - moments.reshape(1, -1)
    cov = phi[test].T @ phi[test] / ntest**2
    cov += phi[train].T @ phi[train] / ntrain**2
    violation_stat = moments.T @ scipy.linalg.pinvh(cov) @ moments

    return violation_stat


def idstrenth_test(Z, X, D, ivreg_gamma, ivreg_zeta, heuristic=False):
    ''' We want to develop a test for whether
        E[D (D - Z'gamma)] = 0
    where gamma is the solution to the IV
        E[X (D - Z'gamma)] = 0
    '''
    nobs = Z.shape[0]
    Dbar = D - Z @ ivreg_gamma.coef_.reshape(-1, 1)
    if heuristic:
        inf_idstrength = D * Dbar - np.mean(D * Dbar)
        der = np.mean(D * Z, axis=0)
        inf_idstrength -= ivreg_gamma.inf_ @ der.reshape(-1, 1)
        idstrength = np.sqrt(nobs) * np.abs(np.mean(D * Dbar))
    else:
        ivreg_zeta.fit(Z, X, -D)
        zeta = ivreg_zeta.coef_.reshape(-1, 1)
        idstrength = np.mean((D + X @ zeta) * Dbar)
        inf_idstrength = (D + X @ zeta) * Dbar - idstrength
        idstrength = np.sqrt(nobs) * np.abs(idstrength)

    idstrength_std = np.sqrt(np.mean(inf_idstrength**2))
    return idstrength, idstrength_std


def weakiv_test(Z, X, D, ivreg_gamma, ivreg_zeta, heuristic=False):
    ''' We want to develop a F-test for whether we have a weak
    instrument, when the instrument is D - Z'gamma and the treatment
    is D. So we want to use
        pi = E[D (D - Z'gamma)] / E[(D - Z'gamma)^2]
    and var_pi being the variance of the estimate of pi,
    where gamma is the solution to the IV
        E[X (D - Z'gamma)] = 0
    '''
    Dbar = D - Z @ ivreg_gamma.coef_.reshape(-1, 1)

    weakiv_pi = np.mean(D * Dbar) / np.mean(Dbar**2)
    inf_pi = Dbar * (D - weakiv_pi * Dbar)

    if heuristic:
        der = - np.mean(Z * (D - weakiv_pi * Dbar), axis=0)
        der += weakiv_pi * np.mean(Dbar * Z, axis=0)
        inf_pi += ivreg_gamma.inf_ @ der.reshape(-1, 1)
    else:
        J = 2 * weakiv_pi * (D - Dbar) - D
        ivreg_zeta.fit(Z, X, J)
        weakiv_zeta = ivreg_zeta.coef_.reshape(-1, 1)
        weakiv_pi = np.mean((D + X @ weakiv_zeta) * Dbar) / np.mean(Dbar**2)
        inf_pi = Dbar * (D - weakiv_pi * Dbar + X @ weakiv_zeta)

    inf_pi = np.mean(Dbar**2)**(-1) * inf_pi

    weakiv_pi_var = np.mean(inf_pi**2) / inf_pi.shape[0]

    return weakiv_pi, weakiv_pi_var


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


class XGBRegressorWrapper(BaseEstimator, RegressorMixin):

    def __init__(self, *, max_depth=3, early_stopping_rounds=50, learning_rate=.1):
        self.max_depth = max_depth
        self.early_stopping_rounds = early_stopping_rounds
        self.learning_rate = learning_rate

    def fit(self, X, y):
        Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=.2)
        self.model_ = XGBRegressor(max_depth=self.max_depth,
                                   early_stopping_rounds=self.early_stopping_rounds,
                                   learning_rate=self.learning_rate, random_state=123)
        self.model_.fit(Xtrain, ytrain, eval_set=[(Xval, yval)], verbose=False)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, *, max_depth=3, early_stopping_rounds=50, learning_rate=.1):
        self.max_depth = max_depth
        self.early_stopping_rounds = early_stopping_rounds
        self.learning_rate = learning_rate

    def fit(self, X, y):
        Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=.2)
        self.model_ = XGBClassifier(max_depth=self.max_depth,
                                   early_stopping_rounds=self.early_stopping_rounds,
                                   learning_rate=self.learning_rate, eval_metric='logloss', random_state=123)
        self.model_.fit(Xtrain, ytrain, eval_set=[(Xval, yval)], verbose=False)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class GridSearchCVList(BaseEstimator):
    """ From `econml.sklearn_extensions.model_selection`

    An extension of GridSearchCV that allows for passing a list of estimators each with their own
    parameter grid and returns the best among all estimators in the list and hyperparameter in their
    corresponding grid. We are only changing the estimator parameter to estimator_list and the param_grid
    parameter to be a list of parameter grids. The rest of the parameters are the same as in
    :meth:`~sklearn.model_selection.GridSearchCV`. See the documentation of that class
    for explanation of the remaining parameters.

    Parameters
    ----------
    estimator_list : list of estimator object.
        Each estimator in th list is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : list of dict or list of list of dictionaries
        For each estimator, the dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    """

    def __init__(self, estimator_list, param_grid_list, scoring=None,
                 n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        self.estimator_list = estimator_list
        self.param_grid_list = param_grid_list
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        return


    def fit(self, X, y=None, **fit_params):
        self._gcv_list = [GridSearchCV(estimator, param_grid, scoring=self.scoring,
                                       n_jobs=self.n_jobs, refit=self.refit, cv=self.cv, verbose=self.verbose,
                                       pre_dispatch=self.pre_dispatch, error_score=self.error_score,
                                       return_train_score=self.return_train_score)
                          for estimator, param_grid in zip(self.estimator_list, self.param_grid_list)]
        self.best_ind_ = np.argmax([gcv.fit(X, y, **fit_params).best_score_ for gcv in self._gcv_list])
        self.best_estimator_ = self._gcv_list[self.best_ind_].best_estimator_
        self.best_score_ = self._gcv_list[self.best_ind_].best_score_
        self.best_params_ = self._gcv_list[self.best_ind_].best_params_
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)