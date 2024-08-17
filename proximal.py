import numpy as np
from sklearn.linear_model import LassoCV, Lasso, MultiTaskLassoCV, MultiTaskLasso, Ridge, RidgeCV
from sklearn.model_selection import check_cv
from sklearn.base import BaseEstimator
import warnings
from crossfit import fit_predict
from ivreg import Regularized2SLS
from joblib import Parallel, delayed
from inference import EmpiricalInferenceResults, InferenceResults
warnings.simplefilter("ignore")


def residualizeW(W, D, Z, X, Y, *, categorical=True,
            cv=5, semi=False, multitask=False, n_jobs=-1, verbose=0,
            random_state=None):
    ''' Residualizes W out of all the other variables using cross-fitting
    and lasso regression models.

    '''
    if len(D.shape) == 1:
        D = D.reshape(-1, 1)
    assert (D.shape[1]==1), "D should be a scalar treatment"
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    #####
    # Residualizing W out of D, Z, X, Y, using cross-fitting (or semi-cross-fitting)
    # and a Lasso model with regularization chosen via cross-validation
    #####
    cv = check_cv(cv, classifier=categorical)
    if hasattr(cv, 'random_state'):
        cv.random_state = random_state
    if multitask:
        model, modelcv = MultiTaskLasso(), MultiTaskLassoCV()
    else:
        model, modelcv = Lasso(), LassoCV()
    splits = list(cv.split(W, D))
    print("Residualizing D...")
    Dres = D - fit_predict(W, D, modelcv, model, splits, semi, multitask, n_jobs, verbose)
    print("Residualizing Z...")
    Zres = Z - fit_predict(W, Z, modelcv, model, splits, semi, multitask, n_jobs, verbose)
    print("Residualizing X...")
    Xres = X - fit_predict(W, X, modelcv, model, splits, semi, multitask, n_jobs, verbose)
    print("Residualizing Y...")
    Yres = Y - fit_predict(W, Y, modelcv, model, splits, semi, multitask, n_jobs, verbose)

    #####
    # Measuring R^2 perfomrance of residualization models (nuisance models)
    #####
    r2D = 1 - np.mean(Dres**2) / np.var(D)
    r2Z = 1 - np.mean(Zres**2) / np.var(Z)
    r2X = 1 - np.mean(Xres**2) / np.var(X)
    r2Y = 1 - np.mean(Yres**2) / np.var(Y)

    return Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y


def estimate_nuisances(Dres, Zres, Xres, Yres, *,
                       cv=5, n_jobs=-1, verbose=0, random_state=None):
    '''
    Estimate regularized nuisance parameters eta and gamma that
    solve the moment restrictions:
      E[(Yres - eta'Xres - c * Dres) (Dres; Zres)] = 0
      E[(Dres - gamma'Zres) Xres] = 0
    as well as a preliminary regularized estimate of the target
    parameter c (with a theoretically invalid standard error).
    Both solutions are estimated using a ridge regularized two-stage
    least squares estimation procedure, with first stage cross-fitting.
    '''
    DZres = np.column_stack([Dres, Zres])
    DXres = np.column_stack([Dres, Xres])
    alphas = np.logspace(-3, 3, 100) * np.sqrt(DZres.shape[0])
    ivreg = Regularized2SLS(modelcv=RidgeCV(fit_intercept=False, alphas=alphas),
                            model=Ridge(fit_intercept=False),
                            cv=cv,
                            n_jobs=n_jobs,
                            verbose=verbose,
                            random_state=random_state)
    ivreg.fit(DZres, DXres, Yres)
    eta = ivreg.point_[1:].reshape(-1, 1)
    point_pre = ivreg.point_[0]
    std_pre = ivreg.stderr_[0]

    alphas = np.logspace(-3, 3, 100) * np.sqrt(Zres.shape[0])
    ivreg = Regularized2SLS(modelcv=RidgeCV(fit_intercept=False, alphas=alphas),
                            model=Ridge(fit_intercept=False),
                            cv=cv,
                            n_jobs=n_jobs,
                            verbose=verbose,
                            random_state=random_state)
    ivreg.fit(Xres, Zres, Dres)
    gamma = ivreg.point_.reshape(-1, 1)

    return eta, gamma, point_pre, std_pre


def estimate_final(Dres, Zres, Xres, Yres, eta, gamma):
    '''
    Final moment solution: solve for c the equation
      E[(Yres - eta'Xres - c * Dres) (Dres - gamma'Zres)] = 0
    '''
    Ybar = Yres - Xres @ eta
    Dbar = Dres - Zres @ gamma
    
    Jdebiased = Dbar.T @ Dres / Dbar.shape[0]
    Jdebiasedinv = 1 / Jdebiased
    point_debiased = Jdebiasedinv @ (Dbar.T @ Ybar / Dbar.shape[0])
    inf = (Ybar - Dres @ point_debiased) * Dbar * Jdebiasedinv
    std_debiased = np.sqrt(np.mean(inf**2) / inf.shape[0])

    return point_debiased[0, 0], std_debiased, np.mean(Jdebiased)


def second_stage(Dres, Zres, Xres, Yres, *,
                 cv=5, n_jobs=-1, verbose=0, random_state=None):
    ''' Estimate nuisance parameters eta and gamma and then estimate
    target parameter using the nuisances.
    '''
    # estimate the nuisance coefficients that solve the moments
    # E[(Yres - eta'Xres - c*Dres) (Dres; Zres)] = 0
    # E[(Dres - gamma'Zres) Xres] = 0
    eta, gamma, point_pre, std_pre = estimate_nuisances(Dres, Zres, Xres, Yres,
                                                        cv=cv, n_jobs=n_jobs, verbose=verbose,
                                                        random_state=random_state)

    # Final moment solution: solve for c the equation
    #   E[(Yres - eta'Xres - c * Dres) (Dres - gamma'Zres)] = 0
    point_debiased, std_debiased, idstrength = estimate_final(Dres, Zres, Xres, Yres,
                                                              eta, gamma)

    return point_debiased, std_debiased, idstrength, point_pre, std_pre, eta, gamma


def proximal_direct_effect(W, D, Z, X, Y, *, categorical=True,
            cv=5, semi=False, multitask=False, n_jobs=-1, verbose=0,
            random_state=None):
    '''
    categorical: whether D is categorical
    cv: fold option for cross-fitting (e.g. number of folds).
        See `sklearn.model_selection.check_cv` for options.
    semi: whether to perform semi-crossfitting (for penalty choice tuning)
    multitask: whether to use multitask models when predicting multivariate targets
    n_jobs: number of jobs for internal parallel loops
    verbose: degree of verbosity
    random_state: random seed for any internal randomness
    '''
    
    # residualize W from all the variables
    Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y = residualizeW(W, D, Z, X, Y,
            categorical=categorical, cv=cv, semi=semi, multitask=multitask,
            n_jobs=n_jobs, verbose=verbose,
            random_state=random_state)

    point_debiased, std_debiased, idstrength, point_pre, std_pre, _, _ = second_stage(
        Dres, Zres, Xres, Yres,
        cv=cv, n_jobs=n_jobs, verbose=verbose, random_state=random_state)

    # reporting point estimate and standard error of Controlled Direct Effect
    # and R^ performance of nuisance models
    return point_debiased, std_debiased, r2D, r2Z, r2X, r2Y, idstrength, point_pre, std_pre


class ProximalDE(BaseEstimator):
    ''' Estimate Controlled Direct Effect using Proximal Causal Inference.

    categorical: whether D is categorical
    cv: fold option for cross-fitting (e.g. number of folds).
        See `sklearn.model_selection.check_cv` for options.
    semi: whether to perform semi-crossfitting (for penalty choice tuning)
    multitask: whether to use multitask models when predicting multivariate targets
    n_jobs: number of jobs for internal parallel loops
    verbose: degree of verbosity
    random_state: random seed for any internal randomness
    '''
    

    def __init__(self, *,
                 categorical=True,
                 cv=5,
                 semi=True,
                 multitask=False,
                 n_jobs=-1,
                 verbose=0,
                 random_state=None):
        self.categorical = categorical
        self.cv = cv
        self.semi = semi
        self.multitask = multitask
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, W, D, Z, X, Y):
        # residualize W from all the variables
        Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y = residualizeW(W, D, Z, X, Y,
                categorical=self.categorical, cv=self.cv, semi=self.semi,
                multitask=self.multitask, n_jobs=self.n_jobs, verbose=self.verbose,
                random_state=self.random_state)

        # estimate the nuisance coefficients that solve the moments
        # E[(Yres - eta'Xres - c*Dres) (Dres; Zres)] = 0
        # E[(Dres - gamma'Zres) Xres] = 0
        eta, gamma, point_pre, std_pre = estimate_nuisances(Dres, Zres, Xres, Yres,
                                                            cv=self.cv, n_jobs=self.n_jobs,
                                                            verbose=self.verbose,
                                                            random_state=self.random_state)

        # Final moment solution: solve for c the equation
        #   E[(Yres - eta'Xres - c * Dres) (Dres - gamma'Zres)] = 0
        point_debiased, std_debiased, idstrength = estimate_final(Dres, Zres, Xres, Yres,
                                                                  eta, gamma)
        
        # Storing fitted parameters and training data as properties of the class
        self.W_ = W
        self.D_ = D
        self.Z_ = Z
        self.X_ = X
        self.Y_ = Y
        self.Dres_ = Dres
        self.Zres_ = Zres
        self.Xres_ = Xres
        self.Yres_ = Yres
        self.r2D_ = r2D
        self.r2Z_ = r2Z
        self.r2X_ = r2X
        self.r2Y_ = r2Y
        self.eta_ = eta
        self.gamma_ = gamma
        self.point_pre_ = point_pre
        self.std_pre_ = std_pre
        self.point_ = point_debiased
        self.std_ = std_debiased
        self.idstrength_ = idstrength

        return self
    
    def summary(self, *, alpha=0.05, value=0):
        return InferenceResults(self.point_, self.std_).summary(alpha=alpha, value=value)

    def subsample_third_stage(self, *,
                              n_subsamples=1000,
                              fraction=.5,
                              replace=False,
                              n_jobs=-1,
                              verbose=0):
        if not hasattr(self, 'Dres_'):
            raise AttributeError("Object is not fitted!")
        
        np.random.seed(self.random_state)
        n = self.Dres_.shape[0]
        nsub = int(np.ceil(n * fraction))
        subsamples = [np.random.choice(n, nsub, replace=replace)
                      for _ in range(n_subsamples)]
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(estimate_final)(self.Dres_[sub],
                                    self.Zres_[sub],
                                    self.Xres_[sub],
                                    self.Yres_[sub],
                                    self.eta_, self.gamma_)
            for sub in subsamples)
        points, _, _ = zip(*results)
        return np.array(points)
    
    def subsample_second_stage(self, *,
                              n_subsamples=1000,
                              fraction=.5,
                              replace=False,
                              n_jobs=-1,
                              verbose=0):
        if not hasattr(self, 'Dres_'):
            raise AttributeError("Object is not fitted!")
        
        np.random.seed(self.random_state)
        n = self.Dres_.shape[0]
        nsub = int(np.ceil(n * fraction))
        subsamples = [np.random.choice(n, nsub, replace=replace)
                      for _ in range(n_subsamples)]
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(second_stage)(self.Dres_[sub],
                                  self.Zres_[sub],
                                  self.Xres_[sub],
                                  self.Yres_[sub],
                                  cv=self.cv, n_jobs=1, verbose=0,
                                  random_state=None)
            for sub in subsamples)
        points, _, _, _, _, _, _ = zip(*results)
        return np.array(points)

    def subsample_all_stages(self, *,
                             n_subsamples=1000,
                             fraction=.5,
                             replace=False,
                             n_jobs=-1,
                             verbose=0):
        if not hasattr(self, 'Dres_'):
            raise AttributeError("Object is not fitted!")
        
        np.random.seed(self.random_state)
        n = self.Dres_.shape[0]
        nsub = int(np.ceil(n * fraction))
        subsamples = [np.random.choice(n, nsub, replace=replace)
                      for _ in range(n_subsamples)]
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(proximal_direct_effect)(self.W_[sub],
                                            self.D_[sub],
                                            self.Z_[sub],
                                            self.X_[sub],
                                            self.Y_[sub],
                                            categorical=self.categorical,
                                            cv=self.cv,
                                            semi=self.semi,
                                            multitask=self.multitask,
                                            n_jobs=1, verbose=0,
                                            random_state=None)
            for sub in subsamples)
        points, _, _, _, _, _, _, _, _ = zip(*results)
        return np.array(points)


    def bootstrap_inference(self, *, stage=3, n_subsamples=1000,
                            fraction=.5, replace=False, n_jobs=-1, verbose=0):
        '''
        stage: one of {1, 2, 3}; whether to bootstrap from first, second or third
            stage of the estimation process. 1 means all process is repeated on
            the sub-sample. 2 means the residualization is not repeated, but the
            rest is. 3 means that neither the residualization nor the estimation
            of nuisance parameters eta and gamma is repeated. Only the final stage.
        '''
        if stage == 3:
            method = self.subsample_third_stage
        elif stage == 2:
            method = self.subsample_second_stage
        elif stage == 1:
            method = self.subsample_all_stages
        else:
            raise AttributeError("Stage should be one of [1, 2, 3]")

        point_dist = method(n_subsamples=n_subsamples,
                            fraction=fraction,
                            replace=replace,
                            n_jobs=n_jobs,
                            verbose=verbose)

        return EmpiricalInferenceResults(self.point_, point_dist)
