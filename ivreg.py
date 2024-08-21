import numpy as np
from sklearn.base import clone, BaseEstimator
from crossfit import fit_predict
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import check_cv

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

        return self
