import numpy as np
from sklearn.linear_model import LassoCV, Lasso, MultiTaskLassoCV, MultiTaskLasso, Ridge, RidgeCV
from sklearn.model_selection import check_cv
from sklearn.base import BaseEstimator
import warnings
from joblib import Parallel, delayed
from statsmodels.iolib.table import SimpleTable
import scipy.stats
from .crossfit import fit_predict
from .ivreg import Regularized2SLS
from .inference import EmpiricalInferenceResults, NormalInferenceResults, pvalue
from .ivtests import weakiv_tests
from .diagnostics import IVDiagnostics
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
        model = MultiTaskLasso(random_state=random_state)
        modelcv = MultiTaskLassoCV(random_state=random_state)
    else:
        model = Lasso(random_state=random_state)
        modelcv = LassoCV(random_state=random_state)

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
    r2D = np.mean(1 - np.mean(Dres**2, axis=0) / np.var(D, axis=0))
    r2Z = np.mean(1 - np.mean(Zres**2, axis=0) / np.var(Z, axis=0))
    r2X = np.mean(1 - np.mean(Xres**2, axis=0) / np.var(X, axis=0))
    r2Y = np.mean(1 - np.mean(Yres**2, axis=0) / np.var(Y, axis=0))

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
    nobs = Dres.shape[0]
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
    eta = ivreg.coef_[1:].reshape(-1, 1)
    point_pre = ivreg.coef_[0]
    std_pre = ivreg.stderr_[0]
    primal_moments = DZres * (Yres - DXres @ ivreg.coef_.reshape(-1, 1))
    primal_violation = np.linalg.norm(np.sqrt(nobs) * np.mean(primal_moments, axis=0) / np.std(primal_moments, axis=0),
                                      ord=np.inf)

    alphas = np.logspace(-3, 3, 100) * np.sqrt(Zres.shape[0])
    ivreg = Regularized2SLS(modelcv=RidgeCV(fit_intercept=False, alphas=alphas),
                            model=Ridge(fit_intercept=False),
                            cv=cv,
                            n_jobs=n_jobs,
                            verbose=verbose,
                            random_state=random_state)
    ivreg.fit(Xres, Zres, Dres)
    gamma = ivreg.coef_.reshape(-1, 1)
    dual_moments = Xres * (Dres - Zres @ gamma)
    dual_violation = np.linalg.norm(np.sqrt(nobs) * np.mean(dual_moments, axis=0) / np.std(dual_moments, axis=0),
                                    ord=np.inf)

    return eta, gamma, point_pre, std_pre, primal_violation, dual_violation


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

    # standardized strength of jacobian that goes into the denominator
    idstrength = np.abs(np.sqrt(Dres.shape[0]) * np.mean(Jdebiased) / np.std(Dbar * Dres))
    
    return point_debiased[0, 0], std_debiased, idstrength, inf.flatten(), Dbar, Ybar


def second_stage(Dres, Zres, Xres, Yres, *,
                 cv=5, n_jobs=-1, verbose=0, random_state=None):
    ''' Estimate nuisance parameters eta and gamma and then estimate
    target parameter using the nuisances.
    '''
    # estimate the nuisance coefficients that are required for the orthogonal moment
    eta, gamma, point_pre, std_pre, _, _ = estimate_nuisances(Dres, Zres, Xres, Yres,
                                                              cv=cv, n_jobs=n_jobs,
                                                              verbose=verbose,
                                                              random_state=random_state)

    # estimate target parameter using the orthogonal moment
    point_debiased, std_debiased, idstrength, inf, Dbar, Ybar = estimate_final(Dres, Zres, Xres, Yres,
                                                                               eta, gamma)

    return point_debiased, std_debiased, idstrength, point_pre, std_pre, eta, gamma, inf, Dbar, Ybar


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

    point_debiased, std_debiased, idstrength, point_pre, std_pre, _, _, _, _, _ = second_stage(
        Dres, Zres, Xres, Yres,
        cv=cv, n_jobs=n_jobs, verbose=verbose, random_state=random_state)

    # reporting point estimate and standard error of Controlled Direct Effect
    # and R^ performance of nuisance models
    return point_debiased, std_debiased, r2D, r2Z, r2X, r2Y, idstrength, point_pre, std_pre


def _gen_subsamples(n, n_subsamples, fraction, replace, random_state):
    np.random.seed(random_state)
    nsub = int(np.ceil(n * fraction))
    return [np.random.choice(n, nsub, replace=replace)
            for _ in range(n_subsamples)]


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
        eta, gamma, point_pre, std_pre, primal_violation, dual_violation = \
            estimate_nuisances(Dres, Zres, Xres, Yres,
                               cv=self.cv, n_jobs=self.n_jobs,
                               verbose=self.verbose,
                               random_state=self.random_state)

        # Final moment solution: solve for c the equation
        #   E[(Yres - eta'Xres - c * Dres) (Dres - gamma'Zres)] = 0
        point_debiased, std_debiased, idstrength, inf, Dbar, Ybar = estimate_final(Dres, Zres, Xres, Yres,
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
        self.primal_violation_ = primal_violation
        self.dual_violation_ = dual_violation
        self.point_ = point_debiased
        self.std_ = std_debiased
        self.idstrength_ = idstrength
        self.inf_ = inf
        self.Dbar_ = Dbar
        self.Ybar_ = Ybar

        return self

    def _check_is_fitted(self):
        if not hasattr(self, 'point_'):
            raise AttributeError("Object is not fitted!")

    def conf_int(self, *, alpha=0.05):
        self._check_is_fitted()
        return NormalInferenceResults(self.point_, self.std_).conf_int(alpha=alpha)

    def robust_conf_int(self, *, lb, ub, ngrid=1000, alpha=0.05):
        ''' Confidence intervals that are robust to weak identification.
        
        lb: lower bound on grid for which to search for feasible solutions
        ub: lower bound on grid for which to search for feasible solutions
        ngrid: number of grid points to search for
        alpha: confidence level
        '''
        self._check_is_fitted()
        grid = np.linspace(lb, ub, ngrid)
        lb, ub = np.inf, -np.inf
        threshold = scipy.stats.chi2.ppf(1 - alpha, df=1)
        for g in grid:
            Ybar = self.Yres_ - self.Xres_ @ self.eta_
            Dbar = self.Dres_ - self.Zres_ @ self.gamma_
            moment = (Ybar - self.Dres_ * g) * Dbar
            stat = moment.shape[0] * np.mean(moment)**2 / np.var(moment)
            if stat < threshold:
                lb = g if lb > g else lb
                ub = g if ub < g else ub
        return lb, ub

    def summary(self, *, alpha=0.05, value=0, decimals=4):
        self._check_is_fitted()
        # target parameter summary
        sm = NormalInferenceResults(self.point_, self.std_).summary(alpha=alpha, value=value)
        # nuisance summary
        res = np.array([self.r2D_, self.r2Z_, self.r2X_, self.r2Y_]).reshape(1, -1)
        res = np.round(res, decimals)
        headers = ['r2D', 'r2Z', 'r2X', 'r2Y']
        sm.tables.append(SimpleTable(res, headers, [0], "R^2 of W-Residual Nuisance Models"))

        # tests for identification and assumption violation
        strength = np.round(self.idstrength_, decimals)
        strength_pval = np.format_float_scientific(pvalue(self.idstrength_), precision=decimals)
        pviolation = np.round(self.primal_violation_, decimals)
        pviolation_pval = np.format_float_scientific(pvalue(self.primal_violation_), precision=decimals)
        dviolation = np.round(self.dual_violation_, decimals)
        dviolation_pval = np.format_float_scientific(pvalue(self.dual_violation_), precision=decimals)
        res = np.array([[strength, pviolation, dviolation],
                        [strength_pval, pviolation_pval, dviolation_pval]]).T
        headers = ['statistic', 'p-value']
        index = ['id_strength^1', 'primal_violation^2', 'dual_violation^3']
        sm.tables.append(SimpleTable(res, headers, index, "Tests for weak ID and moment violation"))

        # identification strength
        sm.add_extra_txt(['With $\\epsilon=\\tilde{Y} - \\tilde{X}^\\top \\eta - \\tilde{D}c$ and $V=\\tilde{D} - \\gamma^\\top \\tilde{Z}$',
                          '1. Identification strength $\\frac{\\sqrt{n} E_n[\\tilde{D} V]}{Std_n(\\tilde{D} V)}$ (ideally above 2): ' + f'{strength}',
                          '2. Maximum violation of primal moments $\\frac{\\sqrt{n} E_n[\\epsilon V]}{Std_n(\\epsilon V)}$ (ideally below 2): ' + f'{pviolation}',
                          '3. Maximum violation of dual moments $\\frac{\\sqrt{n} E_n[V \\tilde{X}]}{Std_n(V \\tilde{X})}$ (ideally below 2): ' + f'{dviolation}'])
        
        # weak IV first stage F-tests
        weak_res = weakiv_tests(self.Dbar_, self.Dres_, self.Ybar_)
        ftest_df1, ftest_df2, Fnonrobust, pnonrobust, Frobust, probust, Feff, Keff, Feff_crit = weak_res
        res = np.array([[np.round(Fnonrobust[0], decimals),
                         np.round(Frobust[0], decimals),
                         np.round(Feff[0], decimals)],
                        [ftest_df1, ftest_df1, 'N/A'],
                        [ftest_df2, ftest_df2, 'N/A'],
                        ['N/A', 'N/A', np.round(Keff[0], decimals)],
                        [np.format_float_scientific(pnonrobust[0], precision=decimals),
                         np.format_float_scientific(probust[0], precision=decimals),
                         'N/A'],
                        ['N/A', 'N/A', np.round(Feff_crit[0], decimals)]]).T
        headers = ['statistic', 'df1', 'df2', 'Keff', 'p-value', 'critical-value']
        index = ['F-nonrobust', 'F-robust', 'F-effective']
        sm.tables.append(SimpleTable(res, headers, index, "Weak IV tests"))

        return sm

    def run_diagnostics(self):
        ''' Returns an ``unusual data'' diagnostics object of type `IVDiagnostics`.
        Can then be used to plot robust statistic diagnostics for the results of
        the estimation process. See `diagnostics.IVDiagnostics` for more details.
        '''
        self._check_is_fitted()
        return IVDiagnostics(add_constant=False).fit(self.Dbar_, self.Dres_, self.Ybar_)

    def subsample_third_stage(self, *,
                              n_subsamples=1000,
                              fraction=.5,
                              replace=False,
                              n_jobs=-1,
                              verbose=0,
                              random_state=None):
        self._check_is_fitted()
        subsamples = _gen_subsamples(self.Dres_.shape[0], n_subsamples,
                                     fraction, replace, random_state)
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(estimate_final)(self.Dres_[sub],
                                    self.Zres_[sub],
                                    self.Xres_[sub],
                                    self.Yres_[sub],
                                    self.eta_, self.gamma_)
            for sub in subsamples)
        # get the distribution of point estimates from the results
        points = next(zip(*results))
        return np.array(points)


    def subsample_second_stage(self, *,
                              n_subsamples=1000,
                              fraction=.5,
                              replace=False,
                              n_jobs=-1,
                              verbose=0,
                              random_state=None):
        self._check_is_fitted()
        subsamples = _gen_subsamples(self.Dres_.shape[0], n_subsamples,
                                     fraction, replace, random_state)
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(second_stage)(self.Dres_[sub],
                                  self.Zres_[sub],
                                  self.Xres_[sub],
                                  self.Yres_[sub],
                                  cv=self.cv, n_jobs=1, verbose=0,
                                  random_state=None)
            for sub in subsamples)
        # get the distribution of point estimates from the results
        points = next(zip(*results))
        return np.array(points)


    def subsample_all_stages(self, *,
                             n_subsamples=1000,
                             fraction=.5,
                             replace=False,
                             n_jobs=-1,
                             verbose=0,
                             random_state=None):
        self._check_is_fitted()
        subsamples = _gen_subsamples(self.Dres_.shape[0], n_subsamples,
                                     fraction, replace, random_state)
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
        # get the distribution of point estimates from the results
        points = next(zip(*results))
        return np.array(points)


    def bootstrap_inference(self, *, stage=3, n_subsamples=1000,
                            fraction=.5, replace=False, n_jobs=-1, verbose=0,
                            random_state=None):
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
                            verbose=verbose,
                            random_state=random_state)

        return EmpiricalInferenceResults(self.point_, point_dist)
