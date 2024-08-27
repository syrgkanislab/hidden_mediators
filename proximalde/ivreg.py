import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import check_cv
from .crossfit import fit_predict
from .utilities import _check_input


class Regularized2SLS(BaseEstimator):
    ''' A general purpose regularized 2SLS estimator that
    also uses cross-fitting when fitting the first stage.

    Parameters
    ----------
    modelcv_first : estimator object
        A (potentially) cross-validated regularized linear regression
        object. If it regularized, then it needs to store the regularization
        penalty in attribute `alpha_` after being fitted.
    model_first : estimator object
        A non cross-validated regularized linear regression object. If
        `semi=False`, this object will be ignored. If `semi=True`, it
        needs to have as input parameter `alpha` that corresponds to the
        penalty level.
    model_final : estimator object
        A (regularized) linear regression object. If it regularized,
        then it needs to store the regularization penalty in attribute
        `alpha_` after being fitted. It also needs to store the final
        parameters in attributes `coef_` and `intercept_`.
        It also needs to have input parameter `fit_intercept`, that
        declares if an intercept was fitted.
    cv : any cross-validation generator option, optional (default=5)
        See `sklearn.model_selection.check_cv` for valid options
    semi : bool, optinal (default=True)
        Whether semi-cross-fitting will be used for the first stage
        regression, or simple cross-fitting.
    multitask : bool, optional (default=False)
        Whether multitasking will be used for the first stage regression
    n_jobs : int or None, optional (default=None)
        Number of jobs for parallelism. `-1` means all cores. `None` means
        no parallelism.
    verbose : int, optional (default=0)
        Degree of verbosity.
    random_state : int or None, optional (default=None)
        Random seed for deterministic behavior or `None` for non-determinism.
    '''

    def __init__(self, *, modelcv_first, model_first, model_final,
                 cv=5,
                 semi=True,
                 multitask=False,
                 n_jobs=None,
                 verbose=0,
                 random_state=None):
        self.model_first = model_first
        self.modelcv_first = modelcv_first
        self.model_final = model_final
        self.cv = cv
        self.semi = semi
        self.multitask = multitask
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, Z, D, Y):
        ''' Fit IV regression

        Parameters
        ----------
        Z : array (n, pz) or (n,)
            Instrument(s)
        D : array (n, pd) or (n,)
            Treatments(s)
        Y : array (n, 1) or (n,)
            Outcome

        Returns
        -------
        self : object
        '''
        Z, D, Y = _check_input(Z, D, Y)
        assert Y.shape[1] == 1, "Y should be scalar!"

        Y = Y.flatten()

        # Getting splits for first stage cross-fitting
        self.cv_ = check_cv(self.cv)
        if hasattr(self.cv_, 'shuffle'):
            self.cv_.shuffle = True
        if hasattr(self.cv_, 'random_state'):
            self.cv_.random_state = self.random_state
        splits = list(self.cv_.split(Z, D))

        # Cross-fitted and regularized first stage predictions
        Q = fit_predict(Z, D, self.modelcv_first, self.model_first,
                        splits, self.semi, self.multitask, self.n_jobs, self.verbose)

        # Regularized second stage
        model_final = clone(self.model_final).fit(Q, Y)

        # Check that `model_final` adheres to the API spec
        if ((not hasattr(model_final, 'coef_')) or
                (not hasattr(model_final, 'intercept_')) or
                (not hasattr(model_final, 'fit_intercept'))):
            raise AttributeError('The object `model_final` needs to have attribtes `coef_` '
                                 'and `intercept_` after being fitted and input parameter '
                                 '`fit_intercept` denoting whether an intercept was fitted.')

        # Parse attributes
        self.coef_ = model_final.coef_
        self.intercept_ = model_final.intercept_

        if hasattr(model_final, 'alpha_'):
            alpha = model_final.alpha_
        else:
            alpha = 0

        # Calculating an approximate standard error, which in theory is
        # invalid. Mostly for comparison and benchmarking purposes.
        if model_final.fit_intercept is True:
            Qc = Q - Q.mean(axis=0, keepdims=True)
            Dc = D - D.mean(axis=0, keepdims=True)
            Yc = Y - Y.mean(axis=0, keepdims=True)
        else:
            Qc, Dc, Yc = Q, D, Y

        J = Qc.T @ Qc / Qc.shape[0] + np.eye(Qc.shape[1]) * (alpha / Qc.shape[0])
        Jinv = np.linalg.pinv(J)
        epsilon = (Yc - Dc @ self.coef_).reshape(-1, 1) * Qc
        inf = epsilon @ Jinv.T
        Cov = inf.T @ inf / inf.shape[0]
        self.stderr_ = np.sqrt(np.diag(Cov) / Qc.shape[0])
        self.inf_ = inf
        # store alternative way of calculating coefficient for
        # testing purposes
        self.coef_alt_ = (Jinv @ (Qc.T @ Yc / Qc.shape[0])).flatten()

        # storing the first stage out-of-fold predictions as attribute
        self.Q_ = Q
        self.alpha_ = alpha

        return self


class RegularizedDualIVSolver(BaseEstimator):
    ''' Finds an l2-regularized solution to the system
        E[QQ'] coef = E[X D]
    using cross-validation based on out-of-sample moment violation
        ||E[(D - coef'Q) X]||_{infty}
    Here X can be thought as an instrument, Q as an endogenous
    treatment and D as an outcome. Calculation assumes that Q, X
    satisfy that E[QX'] = E[QQ'].
    '''
    def __init__(self, *, alphas,
                 cv=5,
                 random_state=None):
        self.alphas = alphas
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, Q, D):
        ''' Fit linear system solution

        Parameters
        ----------
        X : array (n, pz) or (n,)
        Q : array (n, pd) or (n,)
        D : array (n, 1) or (n,)

        Returns
        -------
        self : object
        '''
        X, Q, D = _check_input(X, Q, D)
        assert D.shape[1] == 1, "D should be scalar!"

        # regularized first stage with cross-fitting
        self.cv_ = check_cv(self.cv)
        if hasattr(self.cv_, 'shuffle'):
            self.cv_.shuffle = True
        if hasattr(self.cv_, 'random_state'):
            self.cv_.random_state = self.random_state
        splits = list(self.cv_.split(Q, D))

        alpha_best = 0
        best_violation = np.inf
        for alpha in self.alphas:
            Dbar = np.zeros(D.shape)
            for train, test in splits:
                ntrain = len(train)
                JD = Q[train].T @ Q[train] / Q[train].shape[0]
                JD += np.eye(Q[train].shape[1]) * (alpha / ntrain)
                JDinv = np.linalg.pinv(JD)
                coef = JDinv @ (X[train].T @ D[train] / X[train].shape[0])
                Dbar[test] = D[test] - Q[test] @ coef
            violation = np.linalg.norm(np.mean(Dbar * X, axis=0), ord=np.inf)
            if violation < best_violation:
                best_violation = violation
                alpha_best = alpha

        # Calculate coef using the best penalty choice
        JD = Q.T @ Q / Q.shape[0] + np.eye(Q.shape[1]) * (alpha_best / Q.shape[0])
        JDinv = np.linalg.pinv(JD)
        coef = JDinv @ (X.T @ D / X.shape[0])
        inf = X * (D - Q @ coef).reshape(-1, 1)
        inf = inf @ JDinv.T

        # Storing class attributes
        self.coef_ = coef.flatten()
        self.alpha_ = alpha_best
        self.inf_ = inf

        return self


def advIV(Z, X, Y, alpha):
    n = Z.shape[0]
    XZ = X.T @ Z / n
    ZZinv = np.linalg.pinv(Z.T @ Z / n)
    Q = Z @ ZZinv @ XZ.T
    XX = X.T @ X / n
    QY = Q.T @ Y / n
    XQ = X.T @ Q / n
    Jinv = np.linalg.pinv(XQ + (alpha / n) * XX)
    coef = Jinv @ QY
    inf = Q * Y - (Q + (alpha / n) * X) * (X @ coef)
    inf = inf @ Jinv.T
    return coef, Q, inf


class AdvIV(BaseEstimator):
    ''' Regularized Adversarial IV estimation
    '''
    def __init__(self, *, alphas,
                 cv=5,
                 random_state=None):
        self.alphas = alphas
        self.cv = cv
        self.random_state = random_state

    def fit(self, Z, X, Y):
        ''' Fit linear system solution

        Parameters
        ----------
        Z : array (n, pz) or (n,)
        X : array (n, pd) or (n,)
        Y : array (n, 1) or (n,)

        Returns
        -------
        self : object
        '''
        Z, X, Y = _check_input(Z, X, Y)
        assert Y.shape[1] == 1, "Y should be scalar!"
        nobs = Z.shape[0]

        # regularized first stage with cross-fitting
        self.cv_ = check_cv(self.cv)
        if hasattr(self.cv_, 'shuffle'):
            self.cv_.shuffle = True
        if hasattr(self.cv_, 'random_state'):
            self.cv_.random_state = self.random_state
        splits = list(self.cv_.split(Z, Y))

        mval_best = np.inf
        for alpha in self.alphas:
            moment = np.zeros(Z.shape)
            for train, test in splits:
                coef, _, _ = advIV(Z[train], X[train], Y[train], alpha)
                moment[test] = (Y[test] - X[test] @ coef) * Z[test]
            mval = np.linalg.norm(np.mean(moment, axis=0), ord=2)
            if mval < mval_best:
                mval_best = mval
                alpha_best = alpha

        coef, Q, inf = advIV(Z, X, Y, alpha_best)
        cov = (inf.T @ inf / nobs)
        stderr = np.sqrt(np.diag(cov) / nobs)

        # Storing class attributes
        self.coef_ = coef.flatten()
        self.alpha_ = alpha_best
        self.stderr_ = stderr.flatten()
        self.Q_ = Q
        self.inf_ = inf

        return self
