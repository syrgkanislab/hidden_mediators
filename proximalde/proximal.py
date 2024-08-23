import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import check_cv
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed
from statsmodels.iolib.table import SimpleTable
import scipy.stats
from .crossfit import fit_predict
from .ivreg import Regularized2SLS, RegularizedDualIVSolver
from .inference import EmpiricalInferenceResults, NormalInferenceResults
from .inference import pvalue
from .ivtests import weakiv_tests
from .diagnostics import IVDiagnostics


def _check_input(*args):
    # check that all variables have the same samples
    if len(np.unique([arg.shape[0] for arg in args])) > 1:
        raise AttributeError('All input variables need to have the same number '
                             'of samples')
    # reshape all variables to be 2d matrices
    return (arg.reshape(-1, 1) if len(arg.shape) == 1 else arg for arg in args)


def residualizeW(W, D, Z, X, Y, *, categorical=True,
                 cv=5, semi=False, multitask=False, n_jobs=-1, verbose=0,
                 random_state=None):
    ''' Residualizes W out of all the other variables using cross-fitting
    and lasso regression models.
    '''
    W, D, Z, X, Y = _check_input(W, D, Z, X, Y)

    if D.shape[1] > 1:
        raise AttributeError("D should be a scalar treatment")
    if Y.shape[1] > 1:
        raise AttributeError("Y should be a scalar outcome")

    #####
    # Residualizing W out of D, Z, X, Y, using cross-fitting
    # (or semi-cross-fitting) and a Lasso model with regularization
    # chosen via cross-validation
    #####

    cv = check_cv(cv, y=D, classifier=categorical)
    if hasattr(cv, 'shuffle'):
        cv.shuffle = True
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
    Dres = D - fit_predict(W, D, modelcv, model, splits, semi, multitask,
                           n_jobs, verbose)
    print("Residualizing Z...")
    Zres = Z - fit_predict(W, Z, modelcv, model, splits, semi, multitask,
                           n_jobs, verbose)
    print("Residualizing X...")
    Xres = X - fit_predict(W, X, modelcv, model, splits, semi, multitask,
                           n_jobs, verbose)
    print("Residualizing Y...")
    Yres = Y - fit_predict(W, Y, modelcv, model, splits, semi, multitask,
                           n_jobs, verbose)

    #####
    # Measuring R^2 perfomrance of residualization models (nuisance models)
    #####
    r2D = np.mean(1 - np.mean(Dres**2, axis=0) / np.var(D, axis=0))
    r2Z = np.mean(1 - np.mean(Zres**2, axis=0) / np.var(Z, axis=0))
    r2X = np.mean(1 - np.mean(Xres**2, axis=0) / np.var(X, axis=0))
    r2Y = np.mean(1 - np.mean(Yres**2, axis=0) / np.var(Y, axis=0))

    return Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y, splits


def estimate_nuisances(Dres, Zres, Xres, Yres, *, dual_type='Z',
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
    Returns quantities Dbar = Dres - gamma'Zres and Ybar = Ybar - eta'Xres

    If dual_type='Q', the moment E[(Dres - gamma'Q) X] is used as the
    dual, where Q is the projection of X on (D;Z).
    '''
    Dres, Zres, Xres, Yres = _check_input(Dres, Zres, Xres, Yres)

    if Dres.shape[1] > 1:
        raise AttributeError("D should be a scalar treatment")
    if Yres.shape[1] > 1:
        raise AttributeError("Y should be a scalar outcome")

    nobs = Dres.shape[0]
    DZres = np.column_stack([Dres, Zres])
    DXres = np.column_stack([Dres, Xres])
    alphas = np.logspace(-3, 3, 100)
    ivreg = Regularized2SLS(modelcv_first=RidgeCV(fit_intercept=False,
                                                  alphas=alphas),
                            model_first=Ridge(fit_intercept=False),
                            model_final=RidgeCV(fit_intercept=False,
                                                alphas=alphas),
                            cv=cv,
                            n_jobs=n_jobs,
                            verbose=verbose,
                            random_state=random_state)
    ivreg.fit(DZres, DXres, Yres)
    eta = ivreg.coef_[1:].reshape(-1, 1)
    point_pre = ivreg.coef_[0]
    std_pre = ivreg.stderr_[0]
    primal_moments = DZres * (Yres - DXres @ ivreg.coef_.reshape(-1, 1))
    scaled_primal_moments = np.mean(primal_moments, axis=0)
    scaled_primal_moments /= np.std(primal_moments, axis=0) / np.sqrt(nobs)
    primal_violation = np.linalg.norm(scaled_primal_moments, ord=np.inf)

    # ``outcome'' for the final stage Neyman orthogonal moment
    Ybar = Yres - Xres @ eta

    if dual_type == 'Q':
        Q = ivreg.Q_[:, 1:]  # this is X projected onto D,Z (i.e. best linear predictor of X from D,Z)
        alphas = np.logspace(-3, 3, 100)
        ivreg = RegularizedDualIVSolver(alphas=alphas, cv=cv, random_state=random_state)
        ivreg.fit(Xres, Q, Dres)
        gamma = ivreg.gamma_.reshape(-1, 1)
        # ``instrument'' for the final stage Neyman orthogonal moment
        Dbar = Dres - Q @ gamma
    elif dual_type == 'Z':
        alphas = np.logspace(-3, 3, 100)
        ivreg = Regularized2SLS(modelcv_first=RidgeCV(fit_intercept=False,
                                                      alphas=alphas),
                                model_first=Ridge(fit_intercept=False),
                                model_final=RidgeCV(fit_intercept=False,
                                                    alphas=alphas),
                                cv=cv,
                                n_jobs=n_jobs,
                                verbose=verbose,
                                random_state=random_state)
        ivreg.fit(Xres, Zres, Dres)
        gamma = ivreg.coef_.reshape(-1, 1)
        # ``instrument'' for the final stage Neyman orthogonal moment
        Dbar = Dres - Zres @ gamma
    else:
        raise AttributeError("Unknown `dual_type`. Should be one of {'Q', 'Z'}")

    dual_moments = Xres * Dbar
    scaled_dual_moments = np.mean(dual_moments, axis=0)
    scaled_dual_moments /= np.std(dual_moments, axis=0) / np.sqrt(nobs)
    dual_violation = np.linalg.norm(scaled_dual_moments, ord=np.inf)

    return Dbar, Ybar, eta, gamma, point_pre, std_pre, \
        primal_violation, dual_violation


def estimate_final(Dbar, Dres, Ybar):
    '''
    Final moment solution: solve for c the equation
      E[(Ybar - c * Dres) Dbar] = 0
    '''
    Dbar, Dres, Ybar = _check_input(Dbar, Dres, Ybar)

    if Dbar.shape[1] > 1 or Dres.shape[1] > 1:
        raise AttributeError("D should be a scalar treatment")
    if Ybar.shape[1] > 1:
        raise AttributeError("Y should be a scalar outcome")

    Jdebiased = Dbar.T @ Dres / Dbar.shape[0]
    Jdebiasedinv = 1 / Jdebiased
    point_debiased = Jdebiasedinv @ (Dbar.T @ Ybar / Dbar.shape[0])
    inf = (Ybar - Dres @ point_debiased) * Dbar * Jdebiasedinv
    std_debiased = np.sqrt(np.mean(inf**2) / inf.shape[0])

    # standardized strength of jacobian that goes into the denominator
    idstrength = np.abs(np.mean(Jdebiased))
    idstrength /= np.std(Dbar * Dres) / np.sqrt(Dres.shape[0])

    return point_debiased[0, 0], std_debiased, idstrength, inf.flatten()


def second_stage(Dres, Zres, Xres, Yres, *, dual_type='Z',
                 cv=5, n_jobs=-1, verbose=0, random_state=None):
    ''' Estimate nuisance parameters eta and gamma and then estimate
    target parameter using the nuisances.
    '''
    # estimate the nuisance coefficients that are required
    # for the orthogonal moment
    Dbar, Ybar, eta, gamma, point_pre, std_pre, _, _ = \
        estimate_nuisances(Dres, Zres, Xres, Yres,
                           dual_type=dual_type,
                           cv=cv, n_jobs=n_jobs,
                           verbose=verbose,
                           random_state=random_state)

    # estimate target parameter using the orthogonal moment
    point_debiased, std_debiased, idstrength, inf = estimate_final(Dbar, Dres, Ybar)

    return point_debiased, std_debiased, idstrength, point_pre, std_pre, \
        eta, gamma, inf, Dbar, Ybar


def proximal_direct_effect(W, D, Z, X, Y, *, dual_type='Z', categorical=True,
                           cv=5, semi=True, multitask=False, n_jobs=-1,
                           verbose=0, random_state=None):
    '''
    dual_type: one of {'Z', 'Q'}
        Whether to use E[X (D - gamma'Q)] or E[X (D - gamma'Z)]
        as the dual IV problem to construt the orthogonal instrument Dbar.
    categorical: whether D is categorical
    cv: fold option for cross-fitting (e.g. number of folds).
        See `sklearn.model_selection.check_cv` for options.
    semi: whether to perform semi-crossfitting (for penalty choice tuning)
    multitask: whether to use multitask models when predicting
        multivariate targets
    n_jobs: number of jobs for internal parallel loops
    verbose: degree of verbosity
    random_state: random seed for any internal randomness
    '''
    W, D, Z, X, Y = _check_input(W, D, Z, X, Y)

    if D.shape[1] > 1:
        raise AttributeError("D should be a scalar treatment")
    if Y.shape[1] > 1:
        raise AttributeError("Y should be a scalar outcome")

    # residualize W from all the variables
    Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y, _ = \
        residualizeW(W, D, Z, X, Y,
                     categorical=categorical, cv=cv,
                     semi=semi, multitask=multitask,
                     n_jobs=n_jobs, verbose=verbose,
                     random_state=random_state)

    point_debiased, std_debiased, idstrength, point_pre, std_pre, *_ = \
        second_stage(Dres, Zres, Xres, Yres,
                     dual_type=dual_type,
                     cv=cv, n_jobs=n_jobs, verbose=verbose,
                     random_state=random_state)

    # reporting point estimate and standard error of Controlled Direct Effect
    # and R^ performance of nuisance models
    return point_debiased, std_debiased, r2D, r2Z, r2X, r2Y, \
        idstrength, point_pre, std_pre


def _gen_subsamples(n, n_subsamples, fraction, replace, random_state):
    np.random.seed(random_state)
    nsub = int(np.ceil(n * fraction))
    return [np.random.choice(n, nsub, replace=replace)
            for _ in range(n_subsamples)]


class ProximalDE(BaseEstimator):
    ''' Estimate Controlled Direct Effect using Proximal Causal Inference.

    dual_type: one of {'Z', 'Q'}
        Whether to use E[X (D - gamma'Q)] or E[X (D - gamma'Z)]
        as the dual IV problem to construt the orthogonal instrument Dbar.
    categorical: whether D is categorical
    cv: fold option for cross-fitting (e.g. number of folds).
        See `sklearn.model_selection.check_cv` for options.
    semi: whether to perform semi-crossfitting (for penalty choice tuning)
    multitask: whether to use multitask models when predicting
        multivariate targets
    n_jobs: number of jobs for internal parallel loops
    verbose: degree of verbosity
    random_state: random seed for any internal randomness
    '''

    def __init__(self, *,
                 dual_type='Z',
                 categorical=True,
                 cv=5,
                 semi=True,
                 multitask=False,
                 n_jobs=-1,
                 verbose=0,
                 random_state=None):
        self.dual_type = dual_type
        self.categorical = categorical
        self.cv = cv
        self.semi = semi
        self.multitask = multitask
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, W, D, Z, X, Y):
        '''
        Parameters
        ----------
        W : array (n, pw)
            Controls
        D : array (n, 1) or (n,)
            Treatment
        Z : array (n, pz) or (n,)
            Treatment proxy controls
        X : array (n, px) or (n,)
            Outcome proxy controls
        Y : array (n, 1) or (n,)
            Outcome
        '''
        W, D, Z, X, Y = _check_input(W, D, Z, X, Y)

        if D.shape[1] > 1:
            raise AttributeError("D should be a scalar treatment")
        if Y.shape[1] > 1:
            raise AttributeError("Y should be a scalar outcome")

        # residualize W from all the variables
        Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y, splits = \
            residualizeW(W, D, Z, X, Y,
                         categorical=self.categorical, cv=self.cv,
                         semi=self.semi, multitask=self.multitask,
                         n_jobs=self.n_jobs, verbose=self.verbose,
                         random_state=self.random_state)

        # estimate the nuisance coefficients that solve the moments
        # E[(Yres - eta'Xres - c*Dres) (Dres; Zres)] = 0
        # E[(Dres - gamma'Zres) Xres] = 0
        Dbar, Ybar, eta, gamma, point_pre, std_pre, primal_violation, dual_violation = \
            estimate_nuisances(Dres, Zres, Xres, Yres,
                               dual_type=self.dual_type,
                               cv=self.cv, n_jobs=self.n_jobs,
                               verbose=self.verbose,
                               random_state=self.random_state)

        # Final moment solution: solve for c the equation
        #   E[(Yres - eta'Xres - c * Dres) (Dres - gamma'Zres)] = 0
        point_debiased, std_debiased, idstrength, inf = estimate_final(Dbar, Dres, Ybar)

        # Storing fitted parameters and training data as
        # properties of the class
        self.nobs_ = D.shape[0]
        self.dual_type_ = self.dual_type
        self.categorical_ = self.categorical
        self.cv_ = self.cv
        self.semi_ = self.semi
        self.multitask_ = self.multitask
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
        self.splits_ = splits
        self.Dbar_ = Dbar
        self.Ybar_ = Ybar
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

        return self

    def _check_is_fitted(self):
        if not hasattr(self, 'point_'):
            raise AttributeError("Object is not fitted!")

    def conf_int(self, *, alpha=0.05):
        '''
        Parameters
        ----------
        alpha : float in (0, 1), optional (default=0.05)
            Confidence level of the interval
        '''
        self._check_is_fitted()
        inf = NormalInferenceResults(self.point_, self.std_)
        return inf.conf_int(alpha=alpha)

    def robust_conf_int(self, *, lb, ub, ngrid=1000, alpha=0.05):
        ''' Confidence intervals that are robust to weak identification.

        Parameters
        ----------
        lb : float
            Lower bound on grid for which to search for feasible solutions
        ub : float
            Upper bound on grid for which to search for feasible solutions
        ngrid : int, optional (default=1000)
            Number of grid points to search for
        alpha : float in (0, 1), optional (default=0.05)
            Confidence level of the interval
        '''
        self._check_is_fitted()
        grid = np.linspace(lb, ub, ngrid)
        lb, ub = np.inf, -np.inf
        threshold = scipy.stats.chi2.ppf(1 - alpha, df=1)
        for g in grid:
            moment = (self.Ybar_ - self.Dres_ * g) * self.Dbar_
            stat = moment.shape[0] * np.mean(moment)**2 / np.var(moment)
            if stat < threshold:
                lb = g if lb > g else lb
                ub = g if ub < g else ub
        return lb, ub

    def summary(self, *, alpha=0.05, value=0, decimals=4):
        '''
        Parameters
        ----------
        alpha : float in (0, 1), optional (default=0.05)
            Confidence level of the interval
        value : float, optional (default=0)
            Value to test for hypothesis testing and p-values
        decimals : int, optional (default=4)
            Number of decimal points for floats and precision for scientific formats
        '''
        self._check_is_fitted()

        # target parameter summary
        inf = NormalInferenceResults(self.point_, self.std_)
        sm = inf.summary(alpha=alpha, value=value, decimals=decimals)

        # nuisance summary
        res = np.array([self.r2D_, self.r2Z_, self.r2X_, self.r2Y_])
        res = res.reshape(1, -1)
        res = np.round(res, decimals)
        headers = ['r2D', 'r2Z', 'r2X', 'r2Y']
        sm.tables.append(SimpleTable(res, headers, [0],
                                     "R^2 of W-Residual Nuisance Models"))

        # tests for identification and assumption violation
        strength = np.round(self.idstrength_, decimals)
        strength_pval = np.format_float_scientific(pvalue(self.idstrength_),
                                                   precision=decimals)
        pviolation = np.round(self.primal_violation_, decimals)
        pviolation_pval = pvalue(self.primal_violation_)
        pviolation_pval = np.format_float_scientific(pviolation_pval,
                                                     precision=decimals)
        dviolation = np.round(self.dual_violation_, decimals)
        dviolation_pval = pvalue(self.dual_violation_)
        dviolation_pval = np.format_float_scientific(dviolation_pval,
                                                     precision=decimals)
        res = np.array([[strength, pviolation, dviolation],
                        [strength_pval, pviolation_pval, dviolation_pval]]).T
        headers = ['statistic', 'p-value']
        index = ['id_strength^1', 'primal_violation^2', 'dual_violation^3']
        sm.tables.append(SimpleTable(res, headers, index,
                                     "Tests for weak ID and moment violation"))

        # identification strength
        sm.add_extra_txt([
            'With $\\epsilon=\\tilde{Y} - \\tilde{X}^\\top \\eta - \\tilde{D}c$ '
            'and $V=\\tilde{D} - \\gamma^\\top \\tilde{Z}$',
            '1. Identification strength $\\frac{\\sqrt{n} E_n[\\tilde{D} V]}{Std_n(\\tilde{D} V)}$ '
            '(ideally above 2): ' + f'{strength}',
            '2. Maximum violation of primal moments $\\frac{\\sqrt{n} E_n[\\epsilon V]}{Std_n(\\epsilon V)}$ '
            '(ideally below 2): ' + f'{pviolation}',
            '3. Maximum violation of dual moments $\\frac{\\sqrt{n} E_n[V \\tilde{X}]}{Std_n(V \\tilde{X})}$ '
            '(ideally below 2): ' + f'{dviolation}'])

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
        subsamples = _gen_subsamples(self.nobs_, n_subsamples,
                                     fraction, replace, random_state)
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(estimate_final)(self.Dbar_[sub],
                                    self.Dres_[sub],
                                    self.Ybar_[sub])
            for sub in subsamples)
        # get the distribution of point estimates from the results
        points, *_ = zip(*results)
        return np.array(points), subsamples

    def subsample_second_stage(self, *,
                               n_subsamples=1000,
                               fraction=.5,
                               replace=False,
                               n_jobs=-1,
                               verbose=0,
                               random_state=None):
        self._check_is_fitted()
        subsamples = _gen_subsamples(self.nobs_, n_subsamples,
                                     fraction, replace, random_state)
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(second_stage)(self.Dres_[sub],
                                  self.Zres_[sub],
                                  self.Xres_[sub],
                                  self.Yres_[sub],
                                  dual_type=self.dual_type_,
                                  cv=self.cv_,
                                  n_jobs=1, verbose=0,
                                  random_state=None)
            for sub in subsamples)
        # get the distribution of point estimates from the results
        points, *_ = zip(*results)
        return np.array(points), subsamples

    def subsample_all_stages(self, *,
                             n_subsamples=1000,
                             fraction=.5,
                             replace=False,
                             n_jobs=-1,
                             verbose=0,
                             random_state=None):
        self._check_is_fitted()
        subsamples = _gen_subsamples(self.nobs_, n_subsamples,
                                     fraction, replace, random_state)
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(proximal_direct_effect)(self.W_[sub],
                                            self.D_[sub],
                                            self.Z_[sub],
                                            self.X_[sub],
                                            self.Y_[sub],
                                            dual_type=self.dual_type_,
                                            categorical=self.categorical_,
                                            cv=self.cv_,
                                            semi=self.semi_,
                                            multitask=self.multitask_,
                                            n_jobs=1, verbose=0,
                                            random_state=None)
            for sub in subsamples)
        # get the distribution of point estimates from the results
        points, *_ = zip(*results)
        return np.array(points), subsamples

    def bootstrap_inference(self, *, stage=3, n_subsamples=1000,
                            fraction=.5, replace=False, return_subsamples=False,
                            n_jobs=-1, verbose=0,
                            random_state=None):
        '''
        Parameters
        ----------
        stage : one of {1, 2, 3}, optional (default=3)
            Whether to bootstrap from first, second or third stage of the estimation process.
            1 means all process is repeated on the sub-sample.
            2 means the residualization is not repeated, but the rest is.
            3 means that neither the residualization nor the estimation
            of nuisance parameters eta and gamma is repeated. Only the final stage.
        n_subsamples : int, optional (default=1000)
            Number of subsamples.
        fraction : float in (0, 1), optional (default=.5)
            Size of subsamples as a fraction of the original samples.
        replace : bool, optional (default=False)
            Whether to sample with replacement (True) or without replacement (False)
        return_subsamples : bool, optional (default=False)
            Whether to also return the list of subsample indices apart form the
            bootstrap inference object.
        n_jobs : int or None, optional (default=-1)
            Cores for parallelism. -1 means all cores, None means no parallelism
        verbose : int, optional (default=0)
            Degree of verbosity
        random_state : int or None
            Random seed for determinism

        Returns
        -------
        results : `EmpiricalInferenceResults` object
        subsamples : list of arrays, optional (default not returned)
            The list of subsamples generated and which correspond to
            each point in `results.point_dist`. Only returned if
            `return_subsamples=True`
        '''
        self._check_is_fitted()

        if stage == 3:
            method = self.subsample_third_stage
        elif stage == 2:
            method = self.subsample_second_stage
        elif stage == 1:
            method = self.subsample_all_stages
        else:
            raise AttributeError("Stage should be one of [1, 2, 3]")

        point_dist, subsamples = method(n_subsamples=n_subsamples,
                                        fraction=fraction,
                                        replace=replace,
                                        n_jobs=n_jobs,
                                        verbose=verbose,
                                        random_state=random_state)

        if return_subsamples is True:
            return EmpiricalInferenceResults(self.point_, point_dist), subsamples
        else:
            return EmpiricalInferenceResults(self.point_, point_dist)
