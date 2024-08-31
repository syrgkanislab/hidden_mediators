import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import check_cv
from sklearn.base import BaseEstimator, clone
from joblib import Parallel, delayed
from statsmodels.iolib.table import SimpleTable
import scipy.stats
from .crossfit import fit_predict
from .ivreg import Regularized2SLS, RegularizedDualIVSolver, AdvIV
from .inference import EmpiricalInferenceResults, NormalInferenceResults
from .diagnostics import IVDiagnostics
from .utilities import _check_input


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
    print("Residualizing D...") if verbose > 0 else None
    Dres = D - fit_predict(W, D, modelcv, model, splits, semi, multitask,
                           n_jobs, verbose)
    print("Residualizing Z...") if verbose > 0 else None
    Zres = Z - fit_predict(W, Z, modelcv, model, splits, semi, multitask,
                           n_jobs, verbose)
    print("Residualizing X...") if verbose > 0 else None
    Xres = X - fit_predict(W, X, modelcv, model, splits, semi, multitask,
                           n_jobs, verbose)
    print("Residualizing Y...") if verbose > 0 else None
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


def estimate_nuisances(Dres, Zres, Xres, Yres, *, dual_type='Z', ivreg_type='adv',
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
    # splits for out-of-sample violations
    cv = check_cv(cv)
    if hasattr(cv, 'shuffle'):
        cv.shuffle = True
    if hasattr(cv, 'random_state'):
        cv.random_state = random_state
    splits = list(cv.split(Dres))

    DZres = np.column_stack([Dres, Zres])
    DXres = np.column_stack([Dres, Xres])
    alphas = np.logspace(0, 1, 1) * nobs**(0.4)
    if ivreg_type == '2sls':
        ivreg_eta = Regularized2SLS(modelcv_first=RidgeCV(fit_intercept=False,
                                                          alphas=alphas),
                                    model_first=Ridge(fit_intercept=False),
                                    model_final=RidgeCV(fit_intercept=False,
                                                        alphas=alphas),
                                    cv=cv,
                                    n_jobs=n_jobs,
                                    verbose=verbose,
                                    random_state=random_state)
    elif ivreg_type == 'adv':
        ivreg_eta = AdvIV(alphas=alphas, cv=cv, random_state=random_state)
    else:
        raise AttributeError("Unknown `ivreg_type`. Should be one of {'2sls', 'adv'}")

    # calculate out-of-sample moment violation
    primal_moments = np.zeros(DZres.shape)
    for train, test in splits:
        ivreg_train = clone(ivreg_eta).fit(DZres[train], DXres[train], Yres[train])
        primal_moments[test] = DZres[test] * (Yres[test] - DXres[test] @ ivreg_train.coef_.reshape(-1, 1))
    primal_violation = np.mean(primal_moments, axis=0)
    primal_moments_inf = primal_moments - primal_violation
    primal_violation_cov = primal_moments_inf.T @ primal_moments_inf / nobs
    primal_violation_stat = nobs * primal_violation.T @ np.linalg.pinv(primal_violation_cov) @ primal_violation

    # train on all the data to get coefficient eta
    ivreg_eta.fit(DZres, DXres, Yres)
    eta = ivreg_eta.coef_[1:].reshape(-1, 1)
    point_pre = ivreg_eta.coef_[0]
    std_pre = ivreg_eta.stderr_[0]

    # ``outcome'' for the final stage Neyman orthogonal moment
    Ybar = Yres - Xres @ eta

    if dual_type == 'Q':
        dualIV = ivreg_eta.Q_[:, 1:]  # this is X projected onto D,Z (i.e. best linear predictor of X from D,Z)
        alphas = np.logspace(0, 1, 1) * nobs**(0.4)
        ivreg_gamma = RegularizedDualIVSolver(alphas=alphas, cv=cv, random_state=random_state)
    elif dual_type == 'Z':
        alphas = np.logspace(0, 1, 1) * nobs**(0.4)
        dualIV = Zres
        if ivreg_type == '2sls':
            ivreg_gamma = Regularized2SLS(modelcv_first=RidgeCV(fit_intercept=False,
                                                                alphas=alphas),
                                          model_first=Ridge(fit_intercept=False),
                                          model_final=RidgeCV(fit_intercept=False,
                                                              alphas=alphas),
                                          cv=cv,
                                          n_jobs=n_jobs,
                                          verbose=verbose,
                                          random_state=random_state)
        elif ivreg_type == 'adv':
            ivreg_gamma = AdvIV(alphas=alphas, cv=cv, random_state=random_state)
        else:
            raise AttributeError("Unknown `ivreg_type`. Should be one of {'2sls', 'adv'}")
    else:
        raise AttributeError("Unknown `dual_type`. Should be one of {'Q', 'Z'}")

    # calculate out-of-sample dual moment violation
    Dbar = np.zeros(Dres.shape)
    for train, test in splits:
        ivreg_train = clone(ivreg_gamma).fit(Xres[train], dualIV[train], Dres[train])
        Dbar[test] = Dres[test] - dualIV[test] @ ivreg_train.coef_.reshape(-1, 1)
    dual_moments = Xres * Dbar
    dual_violation = np.mean(dual_moments, axis=0)
    dual_moments_inf = dual_moments - dual_violation
    dual_violation_cov = dual_moments_inf.T @ dual_moments_inf / nobs
    dual_violation_stat = nobs * dual_violation.T @ np.linalg.pinv(dual_violation_cov) @ dual_violation

    # train on all the data to get coefficient gamma
    ivreg_gamma.fit(Xres, dualIV, Dres)
    gamma = ivreg_gamma.coef_.reshape(-1, 1)
    # ``instrument'' for the final stage Neyman orthogonal moment
    Dbar = Dres - dualIV @ gamma

    # standardized strength of jacobian that goes into the denominator
    idstrength = np.sqrt(nobs) * np.abs(np.mean(Dres * Dbar))
    inf_idstrength = Dres * Dbar - np.mean(Dres * Dbar) - np.mean(Dres * dualIV) * ivreg_gamma.inf_
    idstrength_std = np.sqrt(np.mean(inf_idstrength**2))

    return Dbar, Ybar, eta, gamma, point_pre, std_pre, \
        primal_violation_stat, dual_violation_stat, idstrength, idstrength_std, \
        ivreg_eta, ivreg_gamma, dualIV


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

    return point_debiased[0, 0], std_debiased, inf.flatten()


def second_stage(Dres, Zres, Xres, Yres, *, dual_type='Z', ivreg_type='adv',
                 cv=5, n_jobs=-1, verbose=0, random_state=None):
    ''' Estimate nuisance parameters eta and gamma and then estimate
    target parameter using the nuisances.
    '''
    # estimate the nuisance coefficients that are required
    # for the orthogonal moment
    Dbar, Ybar, eta, gamma, point_pre, std_pre, _, _, idstrength, idstrength_std, _, _, _ = \
        estimate_nuisances(Dres, Zres, Xres, Yres,
                           dual_type=dual_type, ivreg_type=ivreg_type,
                           cv=cv, n_jobs=n_jobs,
                           verbose=verbose,
                           random_state=random_state)

    # estimate target parameter using the orthogonal moment
    point_debiased, std_debiased, inf = estimate_final(Dbar, Dres, Ybar)

    return point_debiased, std_debiased, idstrength, idstrength_std, point_pre, std_pre, \
        eta, gamma, inf, Dbar, Ybar


def proximal_direct_effect(W, D, Z, X, Y, *, dual_type='Z', ivreg_type='adv', categorical=True,
                           cv=5, semi=True, multitask=False, n_jobs=-1,
                           verbose=0, random_state=None):
    '''
    dual_type: one of {'Z', 'Q'}
        Whether to use E[X (D - gamma'Q)] or E[X (D - gamma'Z)]
        as the dual IV problem to construt the orthogonal instrument Dbar.
    ivreg_type: on of {'2sls', 'adv'}
        Whether to use regularized 2SLS or regularized adversarial IV to
        solve the l2 regularized IV regressions for the nuisances.
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

    point_debiased, std_debiased, idstrength, idstrength_std, point_pre, std_pre, *_ = \
        second_stage(Dres, Zres, Xres, Yres,
                     dual_type=dual_type, ivreg_type=ivreg_type,
                     cv=cv, n_jobs=n_jobs, verbose=verbose,
                     random_state=random_state)

    # reporting point estimate and standard error of Controlled Direct Effect
    # and R^ performance of nuisance models
    return point_debiased, std_debiased, r2D, r2Z, r2X, r2Y, \
        idstrength, idstrength_std, point_pre, std_pre


def _gen_subsamples(n, n_subsamples, fraction, replace, random_state):
    np.random.seed(random_state)
    nsub = int(np.ceil(n * fraction))
    return [np.random.choice(n, nsub, replace=replace)
            for _ in range(n_subsamples)]


class ProximalDE(BaseEstimator):
    ''' Estimate Controlled Direct Effect using Proximal Causal Inference.

    dual_type: one of {'Z', 'Q'}, optional (default='Z')
        Whether to use E[X (D - gamma'Q)] or E[X (D - gamma'Z)]
        as the dual IV problem to construt the orthogonal instrument Dbar.
    ivreg_type: one of {'2sls', 'adv'}, optional (default='adv')
        Whether to use regularized 2SLS or regularized adversarial IV to
        solve the l2 regularized IV regressions for the nuisances.
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
                 ivreg_type='adv',
                 categorical=True,
                 cv=5,
                 semi=True,
                 multitask=False,
                 n_jobs=-1,
                 verbose=0,
                 random_state=None):
        self.dual_type = dual_type
        self.ivreg_type = ivreg_type
        self.categorical = categorical
        self.cv = cv
        self.semi = semi
        self.multitask = multitask
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, W, D, Z, X, Y):
        ''' Train the estimator

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

        Returns
        -------
        self : object
        '''
        # if diagnostics were previously run after some previous fit then we
        # need to make those diagnostics invalid, since we are refitting
        if hasattr(self, 'diag_'):
            del (self.diag_)

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
        Dbar, Ybar, eta, gamma, point_pre, std_pre, primal_violation, dual_violation, \
            idstrength, idstrength_std, ivreg_eta, ivreg_gamma, dualIV = \
            estimate_nuisances(Dres, Zres, Xres, Yres,
                               dual_type=self.dual_type, ivreg_type=self.ivreg_type,
                               cv=self.cv, n_jobs=self.n_jobs,
                               verbose=self.verbose,
                               random_state=self.random_state)

        # Final moment solution: solve for c the equation
        #   E[(Yres - eta'Xres - c * Dres) (Dres - gamma'Zres)] = 0
        point_debiased, std_debiased, inf = estimate_final(Dbar, Dres, Ybar)

        # Storing fitted parameters and training data as
        # properties of the class
        self.nobs_ = D.shape[0]
        self.pw_ = W.shape[1]
        self.pz_ = Z.shape[1]
        self.px_ = X.shape[1]
        self.dual_type_ = self.dual_type
        self.ivreg_type_ = self.ivreg_type
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
        self.stderr_pre_ = std_pre
        self.primal_violation_ = primal_violation
        self.dual_violation_ = dual_violation
        self.point_ = point_debiased
        self.stderr_ = std_debiased
        self.idstrength_ = idstrength
        self.idstrength_std_ = idstrength_std
        self.ivreg_eta_ = ivreg_eta
        self.ivreg_gamma_ = ivreg_gamma
        self.dualIV_ = dualIV
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

        Returns
        -------
        lb, ub : float, float
            The lower and upper end of the confidence interval
        '''
        self._check_is_fitted()
        inf = NormalInferenceResults(self.point_, self.stderr_)
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

        Returns
        -------
        lb, ub : float, float
            The lower and upper end of the confidence interval
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

    def weakiv_test(self, *, alpha=0.05, tau=0.1, return_pi_and_var=False):
        ''' Simplification of the effective first stage F-test for the case
        of only one instrument. See ivtests.py for more information on these
        tests. See also here:
        https://scholar.harvard.edu/files/stock/files/nbersi2018_methods_lectures_weakiv1-2_v4.pdf
        '''
        self._check_is_fitted()
        pi = np.mean(self.Dres_ * self.Dbar_) / np.mean(self.Dbar_**2)
        inf_pi = self.Dbar_ * (self.Dres_ - pi * self.Dbar_)
        der = - np.mean(self.dualIV_ * (self.Dres_ - pi * self.Dbar_), axis=0)
        der += pi * np.mean(self.Dbar_ * self.dualIV_, axis=0)
        inf_pi += self.ivreg_gamma_.inf_ @ der.reshape(-1, 1)
        inf_pi = np.mean(self.Dbar_**2)**(-1) * inf_pi
        # removed this debiasing as it didn't seem to improve results empirical
        pi = np.mean(inf_pi) + pi  # debiasing point estimate
        var_pi = np.mean(inf_pi**2) / inf_pi.shape[0]
        # moment is E[(D-gamma Z) (D - pi (D - gamma Z))]
        # derivative with gamma is -E[Z (D - pi (D - gamma Z))] + E[(D-gamma Z) * pi * Z]

        if return_pi_and_var:
            return pi**2 / var_pi, scipy.stats.ncx2.ppf(1 - alpha, df=1, nc=1 / tau), pi, var_pi
        else:
            return pi**2 / var_pi, scipy.stats.ncx2.ppf(1 - alpha, df=1, nc=1 / tau)

    def covariance_rank_test(self):
        ''' Singular values of covariance matrix of Xres with Zres.
        If these are all small or there aren't many large, then this is
        a signal of weak proxies. Also the number of non-zero singular
        values, is roughly a bound on the dimensionality of the hidden mediator.
        '''
        _, S, _ = np.linalg.svd(self.Zres_.T @ self.Xres_ / self.nobs_)
        return S

    def summary(self, *, alpha=0.05, tau=0.1, value=0, decimals=4):
        '''
        Parameters
        ----------
        alpha : float in (0, 1), optional (default=0.05)
            Confidence level of the interval
        tau : float in (0, 1), optional (default=0.05)
            Target Nagar bias level that is used in calculating the critical value
            for the weak IV test
        value : float, optional (default=0)
            Value to test for hypothesis testing and p-values
        decimals : int, optional (default=4)
            Number of decimal points for floats and precision for scientific formats
        '''
        self._check_is_fitted()

        # target parameter summary
        inf = NormalInferenceResults(self.point_, self.stderr_)
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
        strength_dist = f'N(0, s={np.round(self.idstrength_std_, decimals)})'
        strength_pval = 'N/A'
        strength_crit = '~ 1'  # np.round(scipy.stats.chi2(1).ppf(1 - alpha), decimals)
        pviolation = np.round(self.primal_violation_, decimals)
        pviolation_dist = f'chi2(df={self.pz_ + 1})'
        pviolation_pval = scipy.stats.chi2(self.pz_ + 1).sf(self.primal_violation_)
        pviolation_pval = np.format_float_scientific(pviolation_pval,
                                                     precision=decimals)
        pviolation_crit = np.round(scipy.stats.chi2(self.pz_ + 1).ppf(1 - alpha), decimals)
        dviolation = np.round(self.dual_violation_, decimals)
        dviolation_dist = f'chi2(df={self.px_})'
        dviolation_pval = scipy.stats.chi2(self.px_).sf(self.dual_violation_)
        dviolation_pval = np.format_float_scientific(dviolation_pval,
                                                     precision=decimals)
        dviolation_crit = np.round(scipy.stats.chi2(self.px_).ppf(1 - alpha), decimals)
        weakiv_stat, weakiv_crit = self.weakiv_test(alpha=alpha, tau=tau)
        weakiv_stat = np.round(weakiv_stat, decimals)
        weakiv_crit = np.round(weakiv_crit, decimals)
        weakiv_pval = 'N/A'
        weakiv_dist = f'chi2nc(df=1, nc={np.round(1/tau, decimals)})'
        res = np.array([[strength, pviolation, dviolation, weakiv_stat],
                        [strength_dist, pviolation_dist, dviolation_dist, weakiv_dist],
                        [strength_pval, pviolation_pval, dviolation_pval, weakiv_pval],
                        [strength_crit, pviolation_crit, dviolation_crit, weakiv_crit],
                        ['statistic > critical', 'statistic < critical',
                         'statistic < critical', 'statistic > critical']]).T
        headers = ['statistic', 'null-distribution', 'p-value', 'critical value', 'ideal']
        index = ['id_strength^1', 'primal_violation^2', 'dual_violation^3', 'weakIV_Ftest^4']
        sm.tables.append(SimpleTable(res, headers, index,
                                     "Tests for weak ID and moment violation"))

        S = self.covariance_rank_test()
        topk = np.min([5, self.px_, self.pz_])
        sm.add_extra_txt([
            f'top-{topk}-singular values of Cov(X, Z): [' + ', '.join([str(np.round(S[i], decimals))
                                                                      for i in range(topk)]) + ']',
            'With $e=\\tilde{Y} - \\tilde{X}^\\top \\eta - \\tilde{D}c$ '
            'and $V=\\tilde{D} - \\gamma^\\top \\tilde{Z}$ and $U = (\\tilde{D};\\tilde{Z})$ '
            'and tilde denoting residual after removing the part predictable from $W$.',
            '1. Identification strength $\\sqrt{n} |E_n[\\tilde{D} V]|$ ',
            'A small statistic implies that the effect is weakly identified because '
            'the instrument V is too weakly correlated with the treatment.',
            'This can be caused if the mediator is very predictable from the treatment.',
            'The std of this strength accounts for the estimation error of the parameter $\\gamma$, '
            'but when `dual_type=Q` it does not account for the estimation error of the projection '
            'matrix that goes into Q. '
            'So in that case the std can potentially be artificially small.',
            '2. Maximum violation of primal moments $n E_n[e U]^\\top E_n[e^2 U U^\\top]^{-1} E_n[e U]$.',
            'Under the null it follows approximately a chi2(dim(z) + 1) distribution',
            'A large violation can occur if X is weakly correlated with the mediator, '
            'but D and Z are correlated with the mediator.',
            '3. Maximum violation of dual moments '
            '$n E_n[V \\tilde{X}]^\\top E_n[V^2 \\tilde{X}\\tilde{X}^\\top]^{-1} E_n[V \\tilde{X}]$ ',
            'Under the null, it follows approximately a chi2(dim(x)) distribution',
            'A large dual violation is a test whether the instrument Z is a weak treatment proxy '
            'and implies weak identification. ',
            'For instance, large violation can occur if Z is weakly correlated with the mediator, '
            'while X and D are correlated with the mediator.',
            '4. Weak IV test with $V$ as the instrument, $\\tilde{D}$ as the treatment '
            'and $\\tilde{Y} - \\tilde{X}^\\top \\eta$ as the outcome. ',
            'It estimates the coefficient $\\hat{\\pi}$ of the first stage regression of $\\tilde{D}$ on $V$ '
            'and uses an estimate of the statistic $\\hat{\\pi}^2 / Var(\\hat{\\pi})$',
            'The test accounts for the estimation error of the parameter $\\gamma$, but when `dual_type=Q` '
            'it does not account for the estimation error of the projection matrix that goes into Q. '
            'So in that case the test can potentially be artificially large.'])

        return sm

    def run_diagnostics(self):
        ''' Returns an ``unusual data'' diagnostics object of type `IVDiagnostics`.
        Can then be used to plot robust statistic diagnostics for the results of
        the estimation process. See `diagnostics.IVDiagnostics` for more details.
        '''
        self._check_is_fitted()
        self.diag_ = IVDiagnostics(add_constant=False).fit(self.Dbar_, self.Dres_, self.Ybar_)
        return self.diag_

    def influential_set(self, max_points=None, alpha=None, use_exact_influence=True,
                        use_robust_conf_inf=False, lb=None, ub=None, ngrid=1000):
        ''' Return a subset of the indices that based on the influence
        functions, if removed, should be able to negate the finding or
        make the confidence interval at the `alpha` level contain zero.
        If `alpha=None`, we return the set that will make the point
        estimate take opposite sign. These results are based on an
        influence function representation and hence the actual impact
        of the influential set should be verified by refitting a new
        clone of the object, after removing the influential set from the
        training data.

        Parameters
        ----------
        max_points : int, optional (default=None)
            If None, then the smallest set that suffices to flip the result
            is returned. Otherwise we return the most `max_points` influential
            points that move the result towards the opposite direction
        alpha : float in (0, 1) or None, optional (default=None)
            The confidence level of the interval, or None if we want
            to overturn sign of the point estimate
        use_exact_influence : bool, optional (default=True)
            Whether to use the exact leave-one-out influence function
            or the asymptotic approximation
        use_robust_conf_int : bool, optional (default=False)
            Whether to use weak-identification-robust confidence intervals
            or normal based intervals
        lb, ub, ngrid : float, float, int, optional (default=None, None, 1000)
            Parameters for the robust confidence interval. Ifngored if
            `use_robust_conf_int=False`. `lb, ub` must be provided if True.

        Returns
        -------
        inds : array
            the indices of the influential set
        '''

        if not hasattr(self, 'diag_'):
            raise AttributeError("Please call the `run_diagnostics` method first.")

        point = self.point_
        if use_exact_influence:
            flat_inf = self.diag_.exact_influence_.flatten()
        else:
            flat_inf = self.diag_.influence_.flatten()

        inds = np.argsort(flat_inf)  # sort samples in increasing influence
        if point <= 0:
            # if point negative, we try to remove points with negative influence
            # value, so that the result becomes positive
            ord_inf = flat_inf[inds]
            neg_infs = ord_inf[ord_inf <= 0]
            # the value we want to overturn is either the point or the upper end
            # of the confidence interval
            if alpha is None:
                ub = point
            elif use_robust_conf_inf is False:
                ub = self.conf_int(alpha=alpha)[1]
            else:
                if ((lb is None) or (ub is None)):
                    raise AttributeError("`lb` and `ub` must be provided for robust interval")
                ub = self.robust_conf_int(lb=lb, ub=ub, ngrid=ngrid, alpha=alpha)[1]
            # we remove the smallest set of points such that the sum of their influence
            # is smaller than this negative value (clipped at zero, in case the confidence
            # interval already covers zero)
            cs = np.cumsum(neg_infs)
            thr = np.argwhere(cs < np.clip(ub, -np.inf, 0))
            # if we didn't manage to find enough sufficient negative influence to overturn
            # then we return the whole set of points
            if len(thr) > 0:
                thr = thr[0, 0]
            else:
                thr = len(neg_infs) - 1
            inds = inds[:thr + 1]
        else:
            # if point positive, we try to remove points with positive influence
            # value, so that the result becomes negative
            ord_inf = flat_inf[inds[::-1]]
            pos_infs = ord_inf[ord_inf >= 0]
            # the value we want to overturn is either the point or the lower end
            # of the confidence interval
            if alpha is None:
                lb = point
            elif use_robust_conf_inf is False:
                lb = self.conf_int(alpha=alpha)[0]
            else:
                if ((lb is None) or (ub is None)):
                    raise AttributeError("`lb` and `ub` must be provided for robust interval")
                lb = self.robust_conf_int(lb=lb, ub=ub, ngrid=ngrid, alpha=alpha)[0]
            # we remove the smallest set of points such that the sum of their influence
            # is larger than this positive value (clipped at zero, in case the confidence
            # interval already covers zero)
            cs = np.cumsum(pos_infs)
            thr = np.argwhere(cs > np.clip(lb, 0, np.inf))
            # if we didn't manage to find enough sufficient positive influence to overturn
            # then we return the whole set of points
            if len(thr) > 0:
                thr = thr[0, 0]
            else:
                thr = len(pos_infs) - 1
            inds = inds[::-1][:thr + 1]

        if max_points is None:
            return inds
        else:
            return inds[:min(max_points, len(inds))]

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
                                  ivreg_type=self.ivreg_type_,
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
                                            ivreg_type=self.ivreg_type_,
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
        ''' Inference and confidence intervals based on subsampling and bootstrap
        sub-sampling at different stages of the estimation process.

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
