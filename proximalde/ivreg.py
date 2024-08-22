import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import check_cv
from .crossfit import fit_predict


class Regularized2SLS(BaseEstimator):
    ''' A general purpose regularized 2SLS estimator that
    also uses cross-fitting when fitting the first stage.
    '''

    def __init__(self, *, modelcv, model,
                 cv=5,
                 semi=True,
                 multitask=False,
                 n_jobs=None,
                 verbose=0,
                 random_state=None):
        self.model = model
        self.modelcv = modelcv
        self.cv = cv
        self.semi = semi
        self.multitask = multitask
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, Z, D, Y):
        Y = Y.flatten()
        # regularized first stage with cross-fitting
        self.cv_ = check_cv(self.cv)
        if hasattr(self.cv_, 'shuffle'):
            self.cv_.shuffle = True
        if hasattr(self.cv_, 'random_state'):
            self.cv_.random_state = self.random_state
        splits = list(self.cv_.split(Z, D))
        Q = fit_predict(Z, D, self.modelcv, self.model,
                        splits, self.semi, self.multitask, self.n_jobs, self.verbose)
        # Regularized second stage
        self.model_final_ = clone(self.modelcv).fit(Q, Y)
        self.coef_ = self.model_final_.coef_

        if hasattr(self.model_final_, 'alpha_'):
            alpha = self.model_final_.alpha_
        else:
            alpha = 0

        # Calculating an approximate standard error, which in theory is
        # invalid. Mostly for comparison and benchmarking purposes.
        J = Q.T @ Q / Q.shape[0] + np.eye(Q.shape[1]) * (alpha / Q.shape[0])
        Jinv = np.linalg.pinv(J)
        epsilon = (Y - D @ self.coef_).reshape(-1, 1) * Q
        Sigma = epsilon.T @ epsilon / epsilon.shape[0]
        Cov = Jinv @ Sigma @ Jinv.T
        self.stderr_ = np.sqrt(np.diag(Cov) / Q.shape[0])

        # storing the first stage out-of-fold predictions as attribute
        self.Q_ = Q

        return self


class RegularizedDualIVSolver(BaseEstimator):
    ''' Finds an l2-regularized solution to the system
        E[QQ'] gamma = E[X D]
    using cross-validation based on out-of-sample moment violation
        ||E[(D - gamma'Q) X]||_{infty}
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
                gamma = JDinv @ (X[train].T @ D[train] / X[train].shape[0])
                Dbar[test] = D[test] - Q[test] @ gamma
            violation = np.linalg.norm(np.mean(Dbar * X, axis=0), ord=np.inf)
            if violation < best_violation:
                best_violation = violation
                alpha_best = alpha

        # Calculate gamma using the best penalty choice
        JD = Q.T @ Q / Q.shape[0] + np.eye(Q.shape[1]) * (alpha_best / Q.shape[0])
        JDinv = np.linalg.pinv(JD)
        gamma = JDinv @ (X.T @ D / X.shape[0])

        # Storing class attributes
        self.gamma_ = gamma.flatten()
        self.alpha_ = alpha_best
