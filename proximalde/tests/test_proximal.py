import numpy as np
import pytest
from scipy.stats import chi2, ncx2
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, MultiTaskLasso
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from ..gen_data import gen_data_complex, gen_data_no_controls, gen_data_no_controls_discrete_m
from ..proximal import residualizeW, estimate_nuisances, estimate_final, \
    second_stage, _gen_subsamples, proximal_direct_effect, ProximalDE
from .utilities import gen_iv_data
from ..ivreg import Regularized2SLS
from ..inference import NormalInferenceResults
from joblib import Parallel, delayed


def test_residualize_w_shapes_and_accuracy():
    ''' Test that the output shapes are correct and
    that when W is random noise, then the R^2 are zero.
    '''
    a, b, c, d, e, f, g = .3, .6, .5, .7, .5, .5, .9
    n = 10000
    pw = 1
    pz, px = 1, 1
    np.random.seed(123)
    W, D, _, Z, X, Y = gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g)
    Wfake = np.random.normal(0, 1, size=(n,))
    Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y, _ = \
        residualizeW(Wfake, D.flatten(), Z.flatten(), X.flatten(), Y.flatten(),
                     categorical=True,
                     cv=5, semi=False, multitask=False, n_jobs=-1, verbose=0,
                     random_state=None)

    assert Dres.shape == (n, 1)
    assert Zres.shape == (n, pz)
    assert Xres.shape == (n, px)
    assert Yres.shape == (n, 1)
    assert np.isclose(r2D, 0, atol=1e-3)
    assert np.isclose(r2Z, 0, atol=1e-3)
    assert np.isclose(r2X, 0, atol=1e-3)
    assert np.isclose(r2Y, 0, atol=1e-3)

    with pytest.raises(AttributeError, match="should be a scalar") as e_info:
        residualizeW(Wfake, np.hstack([D.reshape(-1, 1), D.reshape(-1, 1)]),
                     Z.flatten(), X.flatten(), Y.flatten(),
                     categorical=True,
                     cv=5, semi=False, multitask=False, n_jobs=-1, verbose=0,
                     random_state=None)
    print(e_info)

    with pytest.raises(AttributeError, match="should be a scalar") as e_info:
        residualizeW(Wfake, D, Z.flatten(), X.flatten(),
                     np.hstack([Y.reshape(-1, 1), Y.reshape(-1, 1)]),
                     categorical=True,
                     cv=5, semi=False, multitask=False, n_jobs=-1, verbose=0,
                     random_state=None)
    print(e_info)

    with pytest.raises(AttributeError, match="same number of samples") as e_info:
        residualizeW(Wfake, D[:100], Z.flatten(), X.flatten(), Y,
                     categorical=True,
                     cv=5, semi=False, multitask=False, n_jobs=-1, verbose=0,
                     random_state=None)
    print(e_info)


def test_residualize_w_splitting():
    ''' Test that the splits are correctly generated based on
    the input parameters.
    '''
    a, b, c, d, e, f, g = .3, .6, .5, .7, .5, .5, .9
    n = 100
    pw = 1
    pz, px = 1, 1
    W, D, _, Z, X, Y = gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g)
    _, _, _, _, _, _, _, _, splits = \
        residualizeW(W, D, Z, X, Y, categorical=True,
                     cv=5, semi=False, multitask=False, n_jobs=1, verbose=0,
                     random_state=123)

    splits2 = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(W, D))
    for (tr1, te1), (tr2, te2) in zip(splits, splits2):
        assert np.allclose(tr1, tr2)
        assert np.allclose(te1, te2)

    _, _, _, _, _, _, _, _, splits = \
        residualizeW(W, D, Z, X, Y, categorical=False,
                     cv=5, semi=False, multitask=False, n_jobs=1, verbose=0,
                     random_state=123)

    splits2 = list(KFold(n_splits=5, shuffle=True, random_state=123).split(W, D))
    for (tr1, te1), (tr2, te2) in zip(splits, splits2):
        assert np.allclose(tr1, tr2)
        assert np.allclose(te1, te2)

    _, _, _, _, _, _, _, _, splits = \
        residualizeW(W, W[:, 0], Z, X, Y, categorical=True,
                     cv=5, semi=False, multitask=False, n_jobs=1, verbose=0,
                     random_state=123)

    splits2 = list(KFold(n_splits=5, shuffle=True, random_state=123).split(W, W[:, 0]))
    for (tr1, te1), (tr2, te2) in zip(splits, splits2):
        assert np.allclose(tr1, tr2)
        assert np.allclose(te1, te2)


def test_residualized_w_equivalency():
    ''' Test that the functionality is equivalent to what
    we would expect, using custom code.
    '''
    a, b, c, d, e, f, g = .3, .6, .5, .7, .5, .5, .9
    n = 1000
    pw = 1
    pz, px = 3, 3
    W, D, _, Z, X, Y = gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g)

    Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y, splits = \
        residualizeW(W, D, Z, X, Y, categorical=True,
                     cv=5, semi=False, multitask=False, n_jobs=1, verbose=0,
                     random_state=123)

    for T, resT, r2T in [(D.reshape(-1, 1), Dres, r2D), (Z, Zres, r2Z),
                         (X, Xres, r2X), (Y.reshape(-1, 1), Yres, r2Y)]:
        res = T - np.stack([cross_val_predict(LassoCV(random_state=123), W, T[:, i], cv=splits)
                            for i in range(T.shape[1])], axis=-1)
        assert np.allclose(res, resT)
        assert np.isclose(np.mean(1 - np.mean(res**2, axis=0) / np.var(T, axis=0)), r2T)

    Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y, splits = \
        residualizeW(W, D, Z, X, Y, categorical=True,
                     cv=5, semi=False, multitask=True, n_jobs=1, verbose=0,
                     random_state=123)

    for T, resT, r2T in [(D.reshape(-1, 1), Dres, r2D), (Z, Zres, r2Z),
                         (X, Xres, r2X), (Y.reshape(-1, 1), Yres, r2Y)]:
        res = T - cross_val_predict(MultiTaskLassoCV(random_state=123), W, T, cv=splits)
        assert np.allclose(res, resT)
        assert np.isclose(np.mean(1 - np.mean(res**2, axis=0) / np.var(T, axis=0)), r2T)

    Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y, splits = \
        residualizeW(W, D, Z, X, Y, categorical=True,
                     cv=5, semi=True, multitask=True, n_jobs=1, verbose=0,
                     random_state=123)

    for T, resT, r2T in [(D.reshape(-1, 1), Dres, r2D), (Z, Zres, r2Z),
                         (X, Xres, r2X), (Y.reshape(-1, 1), Yres, r2Y)]:
        alpha = MultiTaskLassoCV(random_state=123).fit(W, T).alpha_
        res = T - cross_val_predict(MultiTaskLasso(alpha=alpha, random_state=123), W, T, cv=splits)
        assert np.allclose(res, resT)
        assert np.isclose(np.mean(1 - np.mean(res**2, axis=0) / np.var(T, axis=0)), r2T)


def test_estimate_nuisances():
    ''' Test that the nuisance parameter function accurately
    recovers the desired coefficients. We pass Z, X, D, Y.
    We create two artificial data generating processes.
    In the first case, we check that eta, point_pre, correspond
    to the IV regression of Y on X, D with instrument Z, D.
    In the second case, we generate IV data, Z, X, Y and we
    pass as X, Z, Y, Y. We check that gamma, correctly recovers
    the coefficient of Y on X, using Z as an instrument.
    '''
    np.random.seed(123)
    n, pz, px, pw = 10000, 3, 3, 1
    alpha_multipliers = np.logspace(0, 2, 3)
    alpha_exponent = 0.35
    Z, X, Y, _ = gen_iv_data(n, pz, px, pw, .5)
    alphas = alpha_multipliers * Z.shape[0]**(alpha_exponent)

    for dual_type in ['Z', 'Q']:
        _, Ybar, eta, gamma, point_pre, std_pre, *_ = estimate_nuisances(Z[:, [0]], Z[:, 1:], X[:, 1:], Y,
                                                                         dual_type=dual_type,
                                                                         ivreg_type='2sls',
                                                                         alpha_multipliers=alpha_multipliers,
                                                                         alpha_exponent=alpha_exponent,
                                                                         cv=2, n_jobs=-1, verbose=0,
                                                                         random_state=123)
        assert np.allclose(Ybar, Y - X[:, 1:] @ eta)
        ivreg = Regularized2SLS(modelcv_first=RidgeCV(fit_intercept=False, alphas=alphas),
                                model_first=Ridge(fit_intercept=False),
                                model_final=RidgeCV(fit_intercept=False, alphas=alphas),
                                cv=2, random_state=123).fit(Z, X, Y)
        coef1 = ivreg.coef_
        assert np.allclose(eta.flatten(), coef1[1:])
        assert np.allclose(point_pre, coef1[0])
        assert np.allclose(std_pre, ivreg.stderr_[0])
        assert np.allclose(point_pre, 1.0, atol=9e-2)
        assert np.allclose(eta.flatten(), np.ones(px) / px, atol=9e-2)

    np.random.seed(123)
    n, pz, px, pw = 10000, 3, 3, 0
    Z, X, Y, _ = gen_iv_data(n, pz, px, pw, .5)
    alphas = alpha_multipliers * Z.shape[0]**(alpha_exponent)

    Dbar, _, eta, gamma, point_pre, std_pre, *_ = estimate_nuisances(Y, X, Z, Y,
                                                                     dual_type='Z',
                                                                     ivreg_type='2sls',
                                                                     alpha_multipliers=alpha_multipliers,
                                                                     alpha_exponent=alpha_exponent,
                                                                     cv=2, n_jobs=-1, verbose=0,
                                                                     random_state=123)
    assert np.allclose(Dbar, Y - X @ gamma)
    ivreg = Regularized2SLS(modelcv_first=RidgeCV(fit_intercept=False, alphas=alphas),
                            model_first=Ridge(fit_intercept=False),
                            model_final=RidgeCV(fit_intercept=False, alphas=alphas),
                            cv=2, random_state=123)
    ivreg.fit(Z, X, Y)
    coef1 = ivreg.coef_
    assert gamma.shape == (px, 1)
    assert np.allclose(gamma.flatten(), coef1)
    assert np.allclose(gamma.flatten(), np.ones(px) / px, atol=5e-2)

    np.random.seed(123)
    n, pz, px, pw = 10000, 2, 2, 0
    Z, X, Y, _ = gen_iv_data(n, pz, px, pw, 1.0)
    alphas = alpha_multipliers * Z.shape[0]**(alpha_exponent)

    Dbar, Ybar, eta, gamma, point_pre, std_pre, *_ = estimate_nuisances(Y, X, X, Y,
                                                                        dual_type='Q',
                                                                        ivreg_type='2sls',
                                                                        alpha_multipliers=alpha_multipliers,
                                                                        alpha_exponent=alpha_exponent,
                                                                        cv=5, n_jobs=-1, verbose=0,
                                                                        random_state=123)
    # assert np.allclose(Dbar, Y - X @ gamma, atol=1e-3)
    ivreg = Regularized2SLS(modelcv_first=RidgeCV(fit_intercept=False, alphas=alphas),
                            model_first=Ridge(fit_intercept=False),
                            model_final=RidgeCV(fit_intercept=False, alphas=alphas),
                            cv=5, random_state=123).fit(X, X, Y)
    coef1 = ivreg.coef_
    assert gamma.shape == (px, 1)
    assert np.allclose(gamma.flatten(), coef1, atol=5e-3)
    assert np.allclose(gamma.flatten(), np.ones(px) / px, atol=5e-2)

    with pytest.raises(AttributeError, match="Unknown `dual_type`") as e_info:
        estimate_nuisances(Y[:100], X[:100], X[:100], Y[:100], dual_type='341324')
    assert e_info.typename == 'AttributeError'

    with pytest.raises(AttributeError, match="Unknown `ivreg_type`") as e_info:
        estimate_nuisances(Y[:100], X[:100], X[:100], Y[:100], ivreg_type='341324')
    assert e_info.typename == 'AttributeError'

    with pytest.raises(AttributeError, match="should be a scalar") as e_info:
        estimate_nuisances(np.hstack([Y, Y]), X, X, Y)
    print(e_info)

    with pytest.raises(AttributeError, match="should be a scalar") as e_info:
        estimate_nuisances(Y, X, X, np.hstack([Y, Y]))
    print(e_info)


def test_violations():
    np.random.seed(123)
    n = 10000
    W, D, _, Z, X, Y = gen_data_no_controls(n, 1, 1, 1, 1., 1., .5, 0.0, 0.0, 1.0, 0.0)
    D = D - D.mean(axis=0, keepdims=True)
    Z = Z - Z.mean(axis=0, keepdims=True)
    X = X - X.mean(axis=0, keepdims=True)
    for dual_type in ['Z', 'Q']:
        _, _, _, _, _, _, pval, dval, strength, strength_std, *_ = estimate_nuisances(D, Z, X, Y,
                                                                                      dual_type=dual_type,
                                                                                      cv=5, n_jobs=-1, verbose=0,
                                                                                      random_state=123)
        print(dual_type, pval, dval, strength / strength_std)
        assert dval > 3.84
        assert pval < 3.84
        assert strength / strength_std > 11.0

    np.random.seed(1234)
    _, D, _, Z, X, Y = gen_data_no_controls(n, 1, 1, 1, 1., 1., .5, 0.0, 1.0, 1.0, 0.0)
    D = D - D.mean(axis=0, keepdims=True)
    Z = Z - Z.mean(axis=0, keepdims=True)
    X = X - X.mean(axis=0, keepdims=True)
    for dual_type in ['Z', 'Q']:
        _, _, _, _, _, _, pval, dval, strength, strength_std, *_ = estimate_nuisances(D, Z, X, Y,
                                                                                      dual_type=dual_type,
                                                                                      cv=5, n_jobs=-1, verbose=0,
                                                                                      random_state=123)
        print(dual_type, pval, dval, strength / strength_std)
        assert dval < 3.84
        assert pval < 3.84
        assert strength > 11

    np.random.seed(123)
    _, D, _, Z, X, Y = gen_data_no_controls(n, 1, 1, 1, 1., 1., .5, 1.0, 0.0, 1.0, 0.0)
    D = D - D.mean(axis=0, keepdims=True)
    Z = Z - Z.mean(axis=0, keepdims=True)
    X = X - X.mean(axis=0, keepdims=True)
    for dual_type in ['Z', 'Q']:
        _, _, _, _, _, _, pval, dval, strength, strength_std, *_ = estimate_nuisances(D, Z, X, Y,
                                                                                      dual_type=dual_type,
                                                                                      cv=5, n_jobs=-1, verbose=0,
                                                                                      random_state=123)
        print(dual_type, pval, dval, strength / strength_std)
        assert pval < 3.84
        assert dval > 3.84 if dual_type == 'Q' else True
        assert strength / strength_std < 11.6 if dual_type == 'Z' else True

    np.random.seed(123)
    _, D, _, Z, X, Y = gen_data_no_controls(n, 1, 1, 1, 1., 1., .5, 0.0, 1.0, 0.0, 0.0)
    D = D - D.mean(axis=0, keepdims=True)
    Z = Z - Z.mean(axis=0, keepdims=True)
    X = X - X.mean(axis=0, keepdims=True)
    for dual_type in ['Z', 'Q']:
        _, _, _, _, _, _, pval, dval, strength, strength_std, *_ = estimate_nuisances(D, Z, X, Y,
                                                                                      dual_type=dual_type,
                                                                                      cv=5, n_jobs=-1, verbose=0,
                                                                                      random_state=123)
        print(dual_type, pval, dval, strength / strength_std)
        assert dval < 3.84
        assert pval > 3.84
        assert strength / strength_std > 11.6


def test_estimate_final():
    ''' Test that the nuisance parameter function accurately
    recovers the desired coefficients. We pass Z, X, Y.
    We verify that the result is the same as 2SLS regression of
    Y on X using Z as the instrument. Verify that the idstrength
    is the same as sqrt{n} * mean(X * Z) / std(X * Z)
    '''
    np.random.seed(123)
    Z, X, Y, _ = gen_iv_data(10000, 1, 1, 0, .5)
    point, std, inf = estimate_final(Z, X, Y)
    ivreg = Regularized2SLS(modelcv_first=LinearRegression(fit_intercept=False),
                            model_first=None,
                            model_final=LinearRegression(fit_intercept=False),
                            semi=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))],
                            random_state=123).fit(Z, X, Y)
    assert np.allclose(point, ivreg.coef_[0])
    assert np.allclose(std, ivreg.stderr_[0])
    assert np.allclose(((Y - X * point) * Z / np.mean(X * Z)).flatten(), inf)
    assert np.allclose(np.std(inf) / np.sqrt(Z.shape[0]), std)
    assert np.allclose(point, 1, atol=2e-2)

    with pytest.raises(AttributeError, match="should be a scalar") as e_info:
        estimate_final(np.hstack([Z, Z]), X, Y)
    print(e_info)

    with pytest.raises(AttributeError, match="should be a scalar") as e_info:
        estimate_final(Z, np.hstack([X, X]), Y)
    print(e_info)

    with pytest.raises(AttributeError, match="should be a scalar") as e_info:
        estimate_final(Z, X, np.hstack([Y, Y]))
    print(e_info)


def test_proximal_de_equivalency():
    ''' Verify that the `proximal_direct_effect` function gives
    the same results as the `ProximalDE` class.
    '''
    np.random.seed(123)
    Z, X, Y, _ = gen_iv_data(10000, 3, 3, 1, .5)
    point, std, *_ = second_stage(Z[:, [0]], Z[:, 1:], X[:, 1:], Y, dual_type='Z', ivreg_type='2sls',
                                  cv=5, n_jobs=-1, verbose=0, random_state=None)

    ivreg = Regularized2SLS(modelcv_first=LinearRegression(fit_intercept=False),
                            model_first=None,
                            model_final=LinearRegression(fit_intercept=False),
                            semi=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))],
                            random_state=123).fit(Z, X, Y)
    assert np.allclose(point, ivreg.coef_[0], atol=1e-3)
    assert np.allclose(std, ivreg.stderr_[0], atol=1e-3)

    np.random.seed(123)
    Z, X, Y, _ = gen_iv_data(10000, 3, 3, 1, .5)
    W = np.random.normal(0, 1, size=(10000, 2))
    ivreg = Regularized2SLS(modelcv_first=LinearRegression(fit_intercept=False),
                            model_first=None,
                            model_final=LinearRegression(fit_intercept=False),
                            semi=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))],
                            random_state=123).fit(Z, X, Y)

    point, std, *_ = proximal_direct_effect(W, Z[:, [0]], Z[:, 1:], X[:, 1:], Y, dual_type='Z', ivreg_type='2sls',
                                            cv=5, n_jobs=-1, verbose=0, random_state=None)
    assert np.allclose(point, ivreg.coef_[0], atol=1e-3)
    assert np.allclose(std, ivreg.stderr_[0], atol=1e-3)

    point, std, *_ = proximal_direct_effect(None, Z[:, [0]], Z[:, 1:], X[:, 1:], Y, dual_type='Z', ivreg_type='2sls',
                                            cv=5, n_jobs=-1, verbose=0, random_state=None)
    assert np.allclose(point, ivreg.coef_[0], atol=1e-3)
    assert np.allclose(std, ivreg.stderr_[0], atol=1e-3)

    with pytest.raises(AttributeError, match="should be a scalar") as e_info:
        proximal_direct_effect(W, Z[:, [0, 1]], Z[:, 1:], X[:, 1:], Y)
    print(e_info)

    with pytest.raises(AttributeError, match="should be a scalar") as e_info:
        proximal_direct_effect(W, Z[:, [0]], Z[:, 1:], X[:, 1:], Z[:, 1:])
    print(e_info)


def test_gen_subsamples():
    s1 = _gen_subsamples(5, 5, .3, False, 123)
    s2 = _gen_subsamples(5, 5, .3, False, 123)
    assert np.all([s1t == s2t for s1t, s2t in zip(s1, s2)])
    assert len(s1) == 5
    assert np.all([len(s1t) == 2 for s1t in s1])
    s1 = _gen_subsamples(5, 10, .2, False, 123)
    assert len(s1) == 10
    assert np.all([len(s1t) == 1 for s1t in s1])
    s1 = _gen_subsamples(10, 20, .6, False, 123)
    assert np.all([len(np.unique(s1t)) == len(s1t) for s1t in s1])
    s1 = _gen_subsamples(10, 20, .6, True, 123)
    assert np.any([len(np.unique(s1t)) < len(s1t) for s1t in s1])

    Z, X, Y, _ = gen_iv_data(100, 1, 1, 0, .5)
    pde = ProximalDE(cv=2, n_jobs=1).fit(Z, Z, Z, X, Y)
    for method in [pde.subsample_third_stage, pde.subsample_second_stage, pde.subsample_all_stages]:
        _, s1 = method(n_subsamples=10, fraction=.3, replace=False, random_state=123)
        _, s2 = method(n_subsamples=10, fraction=.3, replace=False, random_state=123)
        assert np.all([s1t == s2t for s1t, s2t in zip(s1, s2)])
        assert len(s1) == 10
        assert np.all([len(s1t) == 30 for s1t in s1])
        _, s1 = method(n_subsamples=30, fraction=.2, replace=False, random_state=123)
        assert len(s1) == 30
        assert np.all([len(s1t) == 20 for s1t in s1])
        assert np.all([len(np.unique(s1t)) == len(s1t) for s1t in s1])
        _, s1 = method(n_subsamples=30, fraction=.2, replace=True, random_state=123)
        assert np.all([len(s1t) == 20 for s1t in s1])
        assert np.any([len(np.unique(s1t)) < len(s1t) for s1t in s1])

    for stage in [1, 2, 3]:
        _, s1 = pde.bootstrap_inference(stage=stage, n_subsamples=10, fraction=.3,
                                        replace=False, return_subsamples=True, random_state=123)
        _, s2 = pde.bootstrap_inference(stage=stage, n_subsamples=10, fraction=.3,
                                        replace=False, return_subsamples=True, random_state=123)
        assert np.all([s1t == s2t for s1t, s2t in zip(s1, s2)])
        assert len(s1) == 10
        assert np.all([len(s1t) == 30 for s1t in s1])
        _, s1 = pde.bootstrap_inference(stage=stage, n_subsamples=30, fraction=.2,
                                        replace=False, return_subsamples=True, random_state=123)
        assert len(s1) == 30
        assert np.all([len(s1t) == 20 for s1t in s1])
        assert np.all([len(np.unique(s1t)) == len(s1t) for s1t in s1])
        _, s1 = pde.bootstrap_inference(stage=stage, n_subsamples=30, fraction=.2,
                                        replace=True, return_subsamples=True, random_state=123)
        assert np.all([len(s1t) == 20 for s1t in s1])
        assert np.any([len(np.unique(s1t)) < len(s1t) for s1t in s1])


def test_raise_nonfitted():
    ''' Verify that the `ProximalDE` class correctly raises a
    non-fitted error when methods are called before `fit`.
    '''
    pde = ProximalDE(cv=2, n_jobs=1)

    with pytest.raises(AttributeError, match="not fitted") as e_info:
        pde.robust_conf_int(lb=-3, ub=3)
    print(e_info)

    for method in [pde.bootstrap_inference, pde.conf_int, pde.run_diagnostics, pde.subsample_all_stages,
                   pde.subsample_second_stage, pde.subsample_third_stage, pde.summary]:
        with pytest.raises(AttributeError, match="not fitted") as e_info:
            method()
        print(e_info)


def test_pde_fit():
    np.random.seed(123)
    a, b, c, d, e, f, g = .3, .6, .5, .7, .5, .5, .9
    n = 100
    pw = 1
    pz, px = 3, 2
    W, D, _, Z, X, Y = gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g)

    pde = ProximalDE(cv=2, n_jobs=1)
    with pytest.raises(AttributeError, match="should be a scalar") as e_info:
        pde.fit(W, Z[:, [0, 1]], Z[:, 1:], X[:, 1:], Y)
    print(e_info)

    with pytest.raises(AttributeError, match="should be a scalar") as e_info:
        pde.fit(W, Z[:, [0]], Z[:, 1:], X[:, 1:], Z[:, [0, 1]])
    print(e_info)

    for semi, cv, dual_type, ivreg_type, multitask, \
        categorical, random_state in [(True, 2, 'Z', '2sls', True, True, 123),
                                      (False, 3, 'Q', 'adv', False, False, 345)]:
        pde = ProximalDE(dual_type=dual_type,  ivreg_type=ivreg_type, categorical=categorical, cv=cv,
                         semi=semi, multitask=multitask,
                         n_jobs=1, random_state=random_state)
        pde.fit(W, D, Z, X, Y,)
        point, std, r2D, r2Z, r2X, r2Y, \
            idstrength, idstrength_std, point_pre, std_pre = proximal_direct_effect(W, D, Z, X, Y,
                                                                                    dual_type=dual_type,
                                                                                    ivreg_type=ivreg_type,
                                                                                    categorical=categorical,
                                                                                    cv=cv,
                                                                                    semi=semi,
                                                                                    multitask=multitask,
                                                                                    n_jobs=1, verbose=0,
                                                                                    random_state=random_state)
        assert pde.pw_ == pw
        assert pde.pz_ == pz
        assert pde.px_ == px
        assert np.allclose(pde.point_, point)
        assert np.allclose(pde.stderr_, std)
        assert np.allclose(pde.r2D_, r2D)
        assert np.allclose(pde.r2Z_, r2Z)
        assert np.allclose(pde.r2X_, r2X)
        assert np.allclose(pde.r2Y_, r2Y)
        assert np.allclose(pde.idstrength_, idstrength)
        assert np.allclose(pde.idstrength_std_, idstrength_std)
        assert np.allclose(pde.point_pre_, point_pre)
        assert np.allclose(pde.stderr_pre_, std_pre)
        assert pde.nobs_ == n
        assert pde.dual_type_ == dual_type
        assert pde.categorical_ == categorical
        assert pde.cv_ == cv
        assert pde.semi_ == semi
        assert pde.multitask_ == multitask
        assert np.allclose(pde.W_, W)
        assert np.allclose(pde.D_, D.reshape(-1, 1))
        assert np.allclose(pde.Z_, Z)
        assert np.allclose(pde.X_, X)
        assert np.allclose(pde.Y_, Y.reshape(-1, 1))
        assert pde.Dres_.shape == pde.D_.shape
        assert pde.Zres_.shape == pde.Z_.shape
        assert pde.Xres_.shape == pde.X_.shape
        assert pde.Yres_.shape == pde.Y_.shape
        assert pde.Dbar_.shape == pde.D_.shape
        assert pde.Ybar_.shape == pde.Y_.shape
        assert pde.eta_.shape == (X.shape[1], 1)
        assert pde.gamma_.shape == (X.shape[1], 1) if dual_type == 'Q' else (Z.shape[1], 1)
        assert pde.primal_violation_ > 0
        assert pde.dual_violation_ > 0
        assert pde.inf_.shape == (n,)

        if categorical is True:
            splits2 = list(StratifiedKFold(n_splits=cv, shuffle=True,
                                           random_state=random_state).split(W, D.reshape(-1, 1)))
            for (tr1, te1), (tr2, te2) in zip(pde.splits_, splits2):
                assert np.allclose(tr1, tr2)
                assert np.allclose(te1, te2)
        else:
            splits2 = list(KFold(n_splits=cv, shuffle=True,
                                 random_state=random_state).split(W, D.reshape(-1, 1)))
            for (tr1, te1), (tr2, te2) in zip(pde.splits_, splits2):
                assert np.allclose(tr1, tr2)
                assert np.allclose(te1, te2)


def test_pde_conf_int():
    np.random.seed(123)
    a, b, c, d, e, f, g = .3, .6, .5, .7, .5, .5, .9
    n = 5000
    pw = 1
    pz, px = 3, 2
    W, D, _, Z, X, Y = gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g)

    pde = ProximalDE(cv=2, n_jobs=1)
    pde.fit(W, D, Z, X, Y,)
    lb1, ub1 = pde.conf_int(alpha=.1)
    lb2, ub2 = pde.conf_int(alpha=.05)
    assert (lb2 < lb1) & (ub2 > ub1)
    assert (lb2 < .5) & (ub2 > .5)
    lb3, ub3 = pde.robust_conf_int(lb=0, ub=2, ngrid=1000, alpha=.1)
    assert np.isclose(lb3, lb1, atol=5e-3) & np.isclose(ub3, ub1, atol=5e-3)
    lb3, ub3 = pde.robust_conf_int(lb=.5, ub=2, ngrid=1000, alpha=.1)
    assert np.isclose(lb3, .5, atol=5e-3) & np.isclose(ub3, ub1, atol=5e-3)
    lb3, ub3 = pde.robust_conf_int(lb=.5, ub=.55, ngrid=1000, alpha=.1)
    assert np.isclose(lb3, .5, atol=5e-3) & np.isclose(ub3, .55, atol=5e-3)


def test_pde_summary():
    np.random.seed(123)
    a, b, c, d, e, f, g = .3, .6, .5, .7, .5, .5, .9
    n = 5000
    pw = 1
    pz, px = 3, 2
    W, D, _, Z, X, Y = gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g)

    pde = ProximalDE(cv=2, n_jobs=1)
    pde.fit(W, D, Z, X, Y,)

    for alpha, value, decimals in [(.1, 0, 3), (.05, 1, 5)]:
        sm = pde.summary(alpha=alpha, value=value, decimals=decimals).tables
        inf = NormalInferenceResults(pde.point_, pde.stderr_)
        assert sm[0].title == 'Parameter Summary'
        assert sm[0][1][1].data == f'{np.round(pde.point_, decimals)}'
        assert sm[0][1][2].data == f'{np.round(pde.stderr_, decimals)}'
        assert sm[0][1][3].data == f'{np.round(inf.zstat(value=value), decimals)}'
        assert sm[0][1][4].data == f'{np.format_float_scientific(inf.pvalue(value=value), precision=decimals)}'
        assert sm[0][1][5].data == f'{np.round(inf.conf_int(alpha=alpha)[0], decimals)}'
        assert sm[0][1][6].data == f'{np.round(inf.conf_int(alpha=alpha)[1], decimals)}'

        assert f'{sm[1][1][1].data}' == f'{np.round(pde.r2D_, decimals)}'
        assert f'{sm[1][1][2].data}' == f'{np.round(pde.r2Z_, decimals)}'
        assert f'{sm[1][1][3].data}' == f'{np.round(pde.r2X_, decimals)}'
        assert f'{sm[1][1][4].data}' == f'{np.round(pde.r2Y_, decimals)}'

        assert f'{sm[2][1][1].data}' == f'{np.round(pde.idstrength_, decimals)}'
        assert f'{sm[2][2][1].data}' == f'{np.round(pde.primal_violation_, decimals)}'
        assert f'{sm[2][2][3].data}' == \
            f'{np.format_float_scientific(chi2(pz + 1).sf(pde.primal_violation_), precision=decimals)}'
        assert f'{sm[2][3][1].data}' == f'{np.round(pde.dual_violation_, decimals)}'
        assert f'{sm[2][3][3].data}' == \
            f'{np.format_float_scientific(chi2(px).sf(pde.dual_violation_), precision=decimals)}'


def test_pde_run_diagnostics():
    np.random.seed(123)
    a, b, c, d, e, f, g = .3, .6, .5, .7, .5, .5, .9
    n = 100
    pw = 1
    pz, px = 3, 2
    W, D, _, Z, X, Y = gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g)

    pde = ProximalDE(cv=2, n_jobs=1)
    pde.fit(W, D, Z, X, Y)
    diag = pde.run_diagnostics()
    assert diag.dfbeta_.shape == (D.shape[0], 1)
    assert np.allclose(diag.point_, pde.point_)
    assert np.allclose(diag.stderr_, pde.stderr_)


def test_pde_subsample_bootstrap():
    np.random.seed(123)
    a, b, c, d, e, f, g = .3, .6, .5, .7, .5, .5, .9
    n = 100
    pw = 1
    pz, px = 3, 2
    W, D, _, Z, X, Y = gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g)

    pde = ProximalDE(cv=2, n_jobs=1)
    pde.fit(W, D, Z, X, Y)
    for stage in [1, 2, 3]:
        inf = pde.bootstrap_inference(stage=stage, n_subsamples=100)
        lb1, ub1 = inf.conf_int(alpha=.1)
        assert (lb1 < .5) & (ub1 > .5)

    with pytest.raises(AttributeError, match="Unknown `stage`") as e_info:
        pde.bootstrap_inference(stage=4, n_subsamples=100)
    print(e_info)

    pde.fit(None, D, Z, X, Y)
    for stage in [2, 3]:
        inf = pde.bootstrap_inference(stage=stage, n_subsamples=100)
        lb1, ub1 = inf.conf_int(alpha=.1)
        assert (lb1 < .5) & (ub1 > .5)

    with pytest.raises(AttributeError, match="No first stage") as e_info:
        pde.bootstrap_inference(stage=1, n_subsamples=100)
    print(e_info)


def test_influential_set():
    np.random.seed(123)
    for sign in [-1, 1]:
        a, b, c, d, e, f, g = .3, .6, sign * .5, .7, .5, .5, .9
        n = 2000
        pw = 1
        pz, px = 3, 2
        W, D, _, Z, X, Y = gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g)
        est = ProximalDE(random_state=3)
        est.fit(W, D, Z, X, Y)

        if sign == -1:
            assert est.conf_int(alpha=0.05)[1] < 0
        else:
            assert est.conf_int(alpha=0.05)[0] > 0

        with pytest.raises(AttributeError, match="run_diagnostics") as e_info:
            est.influential_set()
        print(e_info)

        est.run_diagnostics()
        assert hasattr(est, 'diag_')
        est.fit(W, D, Z, X, Y)

        with pytest.raises(AttributeError, match="run_diagnostics") as e_info:
            est.influential_set()
        print(e_info)

        est.run_diagnostics()
        inds = est.influential_set(alpha=0.05)
        est2 = clone(est)
        est2.fit(np.delete(W, inds, axis=0), np.delete(D, inds, axis=0),
                 np.delete(Z, inds, axis=0), np.delete(X, inds, axis=0),
                 np.delete(Y, inds, axis=0))

        if sign == -1:
            assert est2.conf_int(alpha=0.05)[1] > 0
        else:
            assert est2.conf_int(alpha=0.05)[0] < 0

        with pytest.raises(AttributeError, match="must be provided") as e_info:
            est.influential_set(alpha=0.05, use_robust_conf_inf=True)
        print(e_info)

        inds = est.influential_set(alpha=0.05, use_robust_conf_inf=True, lb=-2, ub=2)
        est2 = clone(est)
        est2.fit(np.delete(W, inds, axis=0), np.delete(D, inds, axis=0),
                 np.delete(Z, inds, axis=0), np.delete(X, inds, axis=0),
                 np.delete(Y, inds, axis=0))

        if sign == -1:
            assert est2.robust_conf_int(alpha=0.05, lb=-2, ub=2)[1] > 0
        else:
            assert est2.robust_conf_int(alpha=0.05, lb=-2, ub=2)[0] < 0

        inds = est.influential_set(alpha=None)
        est2 = clone(est)
        est2.fit(np.delete(W, inds, axis=0), np.delete(D, inds, axis=0),
                 np.delete(Z, inds, axis=0), np.delete(X, inds, axis=0),
                 np.delete(Y, inds, axis=0))

        if sign == -1:
            assert est2.point_ > 0
        else:
            assert est2.point_ < 0

        inds = est.influential_set(alpha=None, max_points=2000)
        est2 = clone(est)
        est2.fit(np.delete(W, inds, axis=0), np.delete(D, inds, axis=0),
                 np.delete(Z, inds, axis=0), np.delete(X, inds, axis=0),
                 np.delete(Y, inds, axis=0))

        if sign == -1:
            assert est2.point_ > 0
        else:
            assert est2.point_ < 0

        inds = est.influential_set(alpha=None, max_points=4)
        assert len(inds) == 4
        est2 = clone(est)
        est2.fit(np.delete(W, inds, axis=0), np.delete(D, inds, axis=0),
                 np.delete(Z, inds, axis=0), np.delete(X, inds, axis=0),
                 np.delete(Y, inds, axis=0))
        if sign == -1:
            assert est2.point_ > est.point_
        else:
            assert est2.point_ < est.point_

        c = sign * 90
        W, D, _, Z, X, Y = gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g)
        est = ProximalDE(dual_type='Z', cv=3, semi=True,
                         multitask=False, n_jobs=-1, random_state=3, verbose=3)
        est.fit(W, D, Z, X, Y)
        diag = est.run_diagnostics()
        inds = est.influential_set(alpha=0.05)
        assert np.all(diag.exact_influence_[inds] * sign >= 0)
        assert len(inds) == np.sum(diag.exact_influence_ * sign >= 0)
        inds = est.influential_set(alpha=0.05, use_exact_influence=False)
        assert np.all(diag.influence_[inds] * sign >= 0)
        assert len(inds) == np.sum(diag.influence_ * sign >= 0)
        est2 = clone(est)
        est2.fit(np.delete(W, inds, axis=0), np.delete(D, inds, axis=0),
                 np.delete(Z, inds, axis=0), np.delete(X, inds, axis=0),
                 np.delete(Y, inds, axis=0))

        if sign == -1:
            assert est2.conf_int(alpha=0.05)[1] > est.conf_int(alpha=0.05)[1]
        else:
            assert est2.conf_int(alpha=0.05)[0] < est.conf_int(alpha=0.05)[0]


def idstrength_violation_q(sm):
    ''' Tests that when we have weak identification the idstrength statistic
    is small and catches the violation of our assumptions.
    '''
    np.random.seed(123)
    errors = []
    strengths = []
    covs = []
    for _ in range(1):
        n = 100000
        pw = 1
        pz, px = 1, 1
        # Indirect effect is a*b, direct effect is c
        a = .7  # (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
        b = .8  # (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
        c = .5  # (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
        # D has direct relationship to Z, Z has no relationship to M,
        # X has direct relationship to M, X has no direct relationship to Y
        d = (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
        e = (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
        f = (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
        g = (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
        sz, sx, sy = np.random.uniform(.5, 2), np.random.uniform(.5, 2), np.random.uniform(.5, 2)
        W, D, _, Z, X, Y = gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g, sm=sm, sz=sz, sx=sx, sy=sy)
        D = D.reshape(-1, 1)
        D = D - D.mean(axis=0)
        sd = np.sqrt(np.mean(D**2))
        X = X - X.mean(axis=0)
        Z = Z - Z.mean(axis=0)
        Y = Y.reshape(-1, 1)
        Y = Y - Y.mean(axis=0)

        true_Zsq = (e * a + d)**2 * sd**2 + e**2 * sm**2 + sz**2
        print('Z**2', np.mean(Z**2), true_Zsq)
        assert np.allclose(np.mean(Z**2), true_Zsq, atol=5e-2)
        true_Msq = sm**2 + a**2 * sd**2
        true_XZ = f * (e * true_Msq + d * a * sd**2)
        print('X*Z', np.mean(X*Z), true_XZ)
        assert np.allclose(np.mean(X*Z), true_XZ, atol=5e-2)
        true_DZ = (a * e + d) * sd**2
        print('D*Z', np.mean(D * Z), true_DZ)
        assert np.allclose(np.mean(D * Z), true_DZ, atol=5e-2)
        true_DX = a * f * sd**2
        print('D*X', np.mean(D * X), true_DX)
        assert np.allclose(np.mean(D * X), true_DX, atol=5e-2)

        cov = np.array([[sd**2, true_DZ], [true_DZ, true_Zsq]])
        b = np.array([true_DX, true_XZ])
        true_q = np.linalg.inv(cov) @ b
        print('q', LinearRegression().fit(np.hstack([D, Z]), X).coef_, true_q)
        assert np.allclose(LinearRegression().fit(np.hstack([D, Z]), X).coef_, true_q, atol=5e-2)

        # true_Q = np.hstack([D, Z]) @ true_q.reshape(-1, 1)
        # gamma = X*D / X*Q
        true_gamma = true_DX / (true_DX * true_q[0] + true_XZ * true_q[1])
        print("gamma", true_gamma)
        # D^2 - gamma D * Q
        true_strength = sd**2 - true_gamma * (true_q[0] * sd**2 + true_q[1] * true_DZ)
        print("strength", true_strength)

        est = ProximalDE(dual_type='Q', cv=3, semi=True,
                         multitask=False, n_jobs=-1, random_state=3, verbose=0)
        est.fit(W, D, Z, X, Y)
        print('point, std', est.point_, est.stderr_)
        print(est.gamma_, true_gamma)
        assert np.allclose(est.gamma_, true_gamma, rtol=1e-1, atol=8e-2)
        print(np.mean(est.Dres_ * est.Dbar_), true_strength)
        assert np.allclose(np.mean(est.Dres_ * est.Dbar_), true_strength, rtol=1e-1, atol=5e-2)
        print(est.idstrength_, est.primal_violation_, est.dual_violation_)
        cov = (est.point_ - 2 * est.stderr_ <= c) & (est.point_ + 2 * est.stderr_ >= c)
        print(cov, est.idstrength_)
        assert cov or (est.idstrength_ < 1)
        error = np.abs(est.point_ - c)
        assert error < 2e-1 or (est.idstrength_ < 1)
        errors.append(error)
        strengths.append(est.idstrength_)
        covs.append(cov)
    return errors, strengths, covs


def test_strength_violation_q():
    errors = []
    strengths = []
    covs = []
    for sm in np.linspace(0, .5, 10):
        print(sm)
        e, s, c = idstrength_violation_q(sm)
        errors += e
        strengths += s
        covs += c


def idstrength_violation_z(sm):
    np.random.seed(123)
    errors = []
    strengths = []
    covs = []
    for _ in range(1):
        n = 100000
        pw = 1
        pz, px = 1, 1
        # Indirect effect is a*b, direct effect is c
        a = .7  # (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
        b = .8  # (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
        c = .5  # (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
        # D has direct relationship to Z, Z has no relationship to M,
        # X has direct relationship to M, X has no direct relationship to Y
        d = np.random.uniform(.5, 2)
        e = np.random.uniform(.5, 2)
        f = (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
        g = (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
        sz, sx, sy = np.random.uniform(.5, 2), np.random.uniform(.5, 2), np.random.uniform(.5, 2)
        W, D, _, Z, X, Y = gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g, sm=sm, sz=sz, sx=sx, sy=sy)
        D = D.reshape(-1, 1)
        D = D - D.mean(axis=0)
        sd = np.sqrt(np.mean(D**2))
        X = X - X.mean(axis=0)
        Z = Z - Z.mean(axis=0)
        Y = Y.reshape(-1, 1)
        Y = Y - Y.mean(axis=0)

        true_Zsq = (e * a + d)**2 * sd**2 + e**2 * sm**2 + sz**2
        print('Z**2', np.mean(Z**2), true_Zsq)
        assert np.allclose(np.mean(Z**2), true_Zsq, atol=7e-2)
        true_Msq = sm**2 + a**2 * sd**2
        true_XZ = f * (e * true_Msq + d * a * sd**2)
        print('X*Z', np.mean(X*Z), true_XZ)
        assert np.allclose(np.mean(X*Z), true_XZ, atol=7e-2)
        true_DZ = (a * e + d) * sd**2
        print('D*Z', np.mean(D * Z), true_DZ)
        assert np.allclose(np.mean(D * Z), true_DZ, atol=5e-2)
        true_DX = a * f * sd**2
        print('D*X', np.mean(D * X), true_DX)
        assert np.allclose(np.mean(D * X), true_DX, atol=5e-2)
        # D*X / X*Z
        true_gamma = true_DX / true_XZ
        print("gamma", true_gamma)
        # D^2 - gamma D * Z
        true_strength = sd**2 - true_gamma * true_DZ
        print("strength", true_strength)

        est = ProximalDE(dual_type='Z', cv=3, semi=True,
                         multitask=False, n_jobs=-1, random_state=3, verbose=0)
        est.fit(W, D, Z, X, Y)
        print('point, std', est.point_, est.stderr_)
        print(est.gamma_, true_gamma)
        assert np.allclose(est.gamma_, true_gamma, rtol=1e-1, atol=8e-2)
        print(np.mean(est.Dres_ * est.Dbar_), true_strength)
        assert np.allclose(np.mean(est.Dres_ * est.Dbar_), true_strength, rtol=1e-1, atol=5e-2)
        print(est.idstrength_, est.primal_violation_, est.dual_violation_)
        cov = (est.point_ - 2 * est.stderr_ <= c) & (est.point_ + 2 * est.stderr_ >= c)
        print(cov, est.idstrength_)
        assert cov or (est.idstrength_ < 2)
        error = np.abs(est.point_ - c)
        assert (error < .4) or (est.idstrength_ < 2)
        errors.append(error)
        strengths.append(est.idstrength_)
        covs.append(cov)
    return errors, strengths, covs


def test_strength_violation_z():
    errors = []
    strengths = []
    covs = []
    for sm in np.linspace(0, .5, 10):
        print(sm)
        e, s, c = idstrength_violation_z(sm)
        errors += e
        strengths += s
        covs += c


def test_primal_violation_caught_z():
    ''' Test that we catch a violation of the existence of the
    primal solution
    '''
    np.random.seed(1236)
    n = 100000
    pw = 1
    pz, px = 1, 1
    # Indirect effect is a*b, direct effect is c
    a, b, c = .7, .8, .5
    # D has no direct relationship to Z, Z has direct relationship to M,
    # X has no direct relationship to M, X has direct relationship to Y
    d, e, f, g = 0.0, 0.5, 0.0, 1.0
    sm, sz, sx, sy = 2, 1, 1, 1
    W, D, _, Z, X, Y = gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g, sm=sm, sz=sz, sx=sx, sy=sy)
    D = D.reshape(-1, 1)
    D = D - D.mean(axis=0)
    X = X - X.mean(axis=0)
    Z = Z - Z.mean(axis=0)
    Y = Y.reshape(-1, 1)
    Y = Y - Y.mean(axis=0)

    est = ProximalDE(dual_type='Z', cv=3, semi=True,
                     multitask=False, n_jobs=-1, random_state=3, verbose=0)
    est.fit(W, D, Z, X, Y)
    print(est.point_, est.stderr_)
    print(est.idstrength_, est.primal_violation_, est.dual_violation_)
    assert est.idstrength_ > 2
    assert est.primal_violation_ > 3.84
    assert est.dual_violation_ < 4.0


def test_rank_violation_caught():
    ''' Test that we catch a violation of the existence of any
    correlation between X and Z
    '''
    np.random.seed(123)
    n = 500000
    pw = 1
    pz, px = 60, 100
    # Indirect effect is a*b, direct effect is c
    a, b, c = .7, .8, .5
    d, e, f, g = 0.0, 0.0, 1.0, 1.0
    sm, sz, sx, sy = 2, 1, 1, 1
    W, D, _, Z, X, Y = gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g, sm=sm, sz=sz, sx=sx, sy=sy)
    D = D.reshape(-1, 1)
    D = D - D.mean(axis=0)
    X = X - X.mean(axis=0)
    Z = Z - Z.mean(axis=0)
    Y = Y.reshape(-1, 1)
    Y = Y - Y.mean(axis=0)

    est = ProximalDE(dual_type='Z', cv=3, semi=True,
                     multitask=False, n_jobs=-1, random_state=3, verbose=0)
    est.fit(None, D, Z, X, Y)
    with pytest.warns(UserWarning, match="large sample size"):
        svalues, svalues_crit = est.covariance_rank_test(calculate_critical=True)
    assert svalues[0] < svalues_crit

    np.random.seed(123)
    n = 100000
    pw = 1
    pz, px = 10, 20
    # Indirect effect is a*b, direct effect is c
    a, b, c = .7, .8, .5
    d, e, f, g = 0.0, 1.0, 1.0, 1.0
    sm, sz, sx, sy = 2, 1, 1, 1
    W, D, _, Z, X, Y = gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g, sm=sm, sz=sz, sx=sx, sy=sy)
    D = D.reshape(-1, 1)
    D = D - D.mean(axis=0)
    X = X - X.mean(axis=0)
    Z = Z - Z.mean(axis=0)
    Y = Y.reshape(-1, 1)
    Y = Y - Y.mean(axis=0)

    est = ProximalDE(dual_type='Z', cv=3, semi=True,
                     multitask=False, n_jobs=-1, random_state=3, verbose=0)
    est.fit(None, D, Z, X, Y)
    svalues, svalues_crit = est.covariance_rank_test(calculate_critical=True)
    assert svalues[0] > svalues_crit


def test_dual_violation_z():
    ''' Test that we catch a violation of the existence of the
    dual solution
    '''
    np.random.seed(1236)
    n = 100000
    pw = 1
    pz, px = 1, 1
    # Indirect effect is a*b, direct effect is c
    a, b, c = .7, .8, .5
    # D has no direct relationship to Z, Z has no relationship to M,
    # X has direct relationship to M, X has direct relationship to Y
    d, e, f, g = 0.0, 0.0, 1.0, 1.0
    sm, sz, sx, sy = 2, 1, 1, 1
    W, D, _, Z, X, Y = gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g, sm=sm, sz=sz, sx=sx, sy=sy)
    D = D.reshape(-1, 1)
    D = D - D.mean(axis=0)
    X = X - X.mean(axis=0)
    Z = Z - Z.mean(axis=0)
    Y = Y.reshape(-1, 1)
    Y = Y - Y.mean(axis=0)
    est = ProximalDE(dual_type='Z', cv=3, semi=True,
                     multitask=False, n_jobs=-1, random_state=3, verbose=0)
    est.fit(W, D, Z, X, Y)

    print(est.idstrength_, est.primal_violation_, est.dual_violation_)
    assert est.idstrength_ > 2
    assert est.primal_violation_ < 4
    assert est.dual_violation_ > 4


def test_accuracy_no_violations():
    ''' Test that we recover truth when assumptions hold
    '''
    np.random.seed(123)
    for n in [10000, 100000]:
        for pz, px, pass_w in [(3, 2, True), (2, 3, False)]:
            for dual_type, ivreg_type in [('Z', '2sls'), ('Z', 'adv'),
                                          ('Q', '2sls'), ('Q', 'adv')]:
                for _ in range(5):
                    print(n, pz, px, dual_type, ivreg_type)
                    pw = 1
                    # Indirect effect is a*b, direct effect is c
                    a = .7  # (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
                    b = .8  # (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
                    c = (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
                    # D has direct relationship to Z, Z has no relationship to M,
                    # X has direct relationship to M, X has no direct relationship to Y
                    d = np.random.uniform(.5, 2)
                    e = np.random.uniform(.5, 2)
                    f = (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
                    g = (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
                    sm = np.random.uniform(.5, 2)
                    sz, sx, sy = np.random.uniform(.5, 2), np.random.uniform(.5, 2), np.random.uniform(.5, 2)
                    W, D, _, Z, X, Y = gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g,
                                                            sm=sm, sz=sz, sx=sx, sy=sy)

                    if pass_w:
                        est = ProximalDE(dual_type=dual_type, ivreg_type=ivreg_type, cv=3, semi=True,
                                         multitask=False, n_jobs=-1, random_state=3, verbose=0)
                        est.fit(W, D, Z, X, Y)
                    else:
                        est = ProximalDE(dual_type=dual_type, ivreg_type=ivreg_type,
                                         alpha_multipliers=np.array([1.0, 1.0 * n]),
                                         alpha_exponent=0.39,
                                         cv=3, semi=True,
                                         multitask=False, n_jobs=-1, random_state=3, verbose=0)
                        est.fit(None, D, Z, X, Y)
                        assert np.allclose(est.alpha_multipliers_, np.array([1.0, 1.0 * n]))
                        assert est.alpha_exponent_ == 0.39
                        assert est.ivreg_eta_.alpha_ == n**(0.39)
                        assert est.ivreg_gamma_.alpha_ == n**(.39)

                    print(c, est.point_, est.stderr_)
                    cov = (est.point_ - 4 * est.stderr_ <= c) & (est.point_ + 4 * est.stderr_ >= c)
                    print(cov, est.idstrength_)
                    assert cov and (est.idstrength_ > 2)
                    error = np.abs(est.point_ - c)
                    assert (error < .2) and (est.idstrength_ > 2)


def test_weakiv_tests():
    np.random.seed(12)
    for n in [10000, 100000]:
        for pz in [1, 2]:
            for px in [1, 2]:
                for sm in [0.01, 2.0]:
                    for dual_type in ['Z']:
                        for ivreg_type in ['2sls', 'adv']:
                            print('n, pz, px, sm, dual_type, ivreg_type = ', n, pz, px, sm, dual_type, ivreg_type)
                            pw = 1
                            # Indirect effect is a*b, direct effect is c
                            a = .7  # (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
                            b = .8  # (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
                            c = .5  # (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
                            # D has direct relationship to Z, Z has no relationship to M,
                            # X has direct relationship to M, X has no direct relationship to Y
                            d = 0  # np.random.uniform(.5, 2)
                            e = 1  # np.random.uniform(.5, 2)
                            f = (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
                            g = (2 * np.random.binomial(1, .5) - 1) * np.random.uniform(.5, 2)
                            sz, sx, sy = np.random.uniform(.5, 2), np.random.uniform(.5, 2), np.random.uniform(.5, 2)
                            W, D, _, Z, X, Y = gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g,
                                                                    sm=sm, sz=sz, sx=sx, sy=sy)
                            D = D.reshape(-1, 1)
                            D = D - D.mean(axis=0)
                            X = X - X.mean(axis=0)
                            Z = Z - Z.mean(axis=0)
                            Y = Y.reshape(-1, 1)
                            Y = Y - Y.mean(axis=0)
                            est = ProximalDE(dual_type=dual_type, ivreg_type=ivreg_type, cv=5, semi=True,
                                             multitask=False, n_jobs=-1, random_state=3, verbose=0)
                            est.fit(W, D, Z, X, Y)
                            if pz == 1:
                                weakiv_stat, _, _, weakiv_crit, pi, var_pi = est.weakiv_test(alpha=0.06, tau=0.099,
                                                                                             return_pi_and_var=True,
                                                                                             decimals=20)
                                assert np.isscalar(pi)
                                assert np.isscalar(var_pi)
                            else:
                                weakiv_stat, _, _, weakiv_crit = est.weakiv_test(alpha=0.06, tau=0.099,
                                                                                 return_pi_and_var=False,
                                                                                 decimals=20)
                            if est.dualIV_.shape[1] == 1:
                                pi = np.mean(est.Dres_ * est.Dbar_) / np.mean(est.Dbar_**2)
                                inf_pi = est.Dbar_ * (est.Dres_ - pi * est.Dbar_)
                                der = - np.mean(est.dualIV_ * (est.Dres_ - pi * est.Dbar_))
                                der += np.mean(est.Dbar_ * pi * est.dualIV_)
                                inf_pi += der * est.ivreg_gamma_.inf_
                                inf_pi = np.mean(est.Dbar_**2)**(-1) * inf_pi
                                pi = pi + np.mean(inf_pi)
                                cov_pi = np.mean(inf_pi**2) / inf_pi.shape[0]
                                assert np.allclose(pi**2 / cov_pi, weakiv_stat)
                                assert np.allclose(weakiv_crit, ncx2.ppf(1 - 0.06, df=1, nc=1/0.099))

                            print(est.point_, est.stderr_, weakiv_stat, weakiv_crit)
                            if sm <= 0.1:
                                assert weakiv_stat < weakiv_crit
                            if sm >= 0.2:
                                assert weakiv_stat > weakiv_crit


def true_params(pw, pz, px, a, b, c, d, e, f, g, sm, sz=1.0, sd=0.5):
    '''true parameters for gamma and strenght for 1d Z and X'''
    if pz == 1 and px == 1:
        # true_Zsq = (e * a + d)**2 * sd**2 + e**2 * sm**2 + sz**2
        true_Msq = sm**2 + a**2 * sd**2
        true_XZ = f * (e * true_Msq + d * a * sd**2)
        true_DZ = (a * e + d) * sd**2
        true_DX = a * f * sd**2
        # D*X / X*Z
        true_gamma = true_DX / true_XZ
        # D^2 - gamma D * Z
        true_strength = sd**2 - true_gamma * true_DZ
        return true_gamma, true_strength
    else:
        raise AttributeError("Not available")


def exp_summary(it, n, pw, pz, px, a, b, c, d, e, f, g, sm):
    np.random.seed(it)
    W, D, _, Z, X, Y = gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g, sm=sm)
    est = ProximalDE(dual_type='Z', cv=3, semi=True,
                     multitask=False, n_jobs=1, random_state=3, verbose=0)
    est.fit(W, D, Z, X, Y)
    lb, ub = est.robust_conf_int(lb=-2, ub=2)
    weakiv_stat, _, _, _, pi, var_pi = est.weakiv_test(return_pi_and_var=True)
    eigs, _ = est.covariance_rank_test(calculate_critical=True)
    maxeig = eigs[0]
    return est.stderr_, est.idstrength_, est.primal_violation_, est.dual_violation_, est.point_, lb, ub, \
        weakiv_stat, maxeig, pi, var_pi


def test_pi_and_var_pi():
    np.random.seed(123)
    n = 10000
    pw = 1
    pz, px = 1, 1
    # Indirect effect is a*b, direct effect is c
    a, b, c = 1.0, 1.0, .5
    d, e, f, g = 1.0, 1.0, 1.0, 1.0
    sm = 2.0
    res = np.array(Parallel(n_jobs=-1, verbose=3)(delayed(exp_summary)(it, n, pw, pz, px, a, b, c, d, e, f, g, sm)
                                                  for it in range(100)))

    pi = res[:, 9]
    var_pi = n * res[:, 10]
    true_gamma, _ = true_params(pw, pz, px, a, b, c, d, e, f, g, sm)
    W, D, _, Z, X, Y = gen_data_no_controls(100000, pw, pz, px, a, b, c, d, e, f, g, sm=sm)
    D = D.reshape(-1, 1)
    D = D - D.mean(axis=0)
    X = X - X.mean(axis=0)
    Z = Z - Z.mean(axis=0)
    true_pi = LinearRegression(fit_intercept=False).fit(D - true_gamma * Z, D.flatten()).coef_[0]
    print(np.mean(pi), true_pi)
    assert np.allclose(np.mean(pi), true_pi, atol=5e-3)
    print(n * np.var(pi), np.mean(var_pi))
    assert np.allclose(n * np.var(pi), np.mean(var_pi), atol=5e-3)
    print(n * np.var(pi), np.percentile(var_pi, 1))
    assert np.allclose(n * np.var(pi), np.percentile(var_pi, 1), atol=5e-3)
    print(n * np.var(pi), np.percentile(var_pi, 99))
    assert np.allclose(n * np.var(pi), np.percentile(var_pi, 99), atol=5e-3)


def exp_summary_multidim(it, n, pm, pw, pz, px, a, b, c, d, E, F, g):
    np.random.seed(it)
    _, D, _, Z, X, Y = gen_data_no_controls_discrete_m(n, pw, pz, px, a, b, c, d, E, F, g, pm=pm)
    est = ProximalDE(dual_type='Z', cv=3, semi=True,
                     multitask=False, n_jobs=1, random_state=3, verbose=0)
    est.fit(None, D, Z, X, Y)
    return est.primal_violation_, est.dual_violation_


def test_multidim_mediator_violations_nominal_failure_prob():
    np.random.seed(123)
    pw = 1
    pm = 7
    for n, pz, px in [(10000, 20, 10), (10000, 80, 50)]:
        # Indirect effect is a*b, direct effect is c
        a, b, c = 1.0, 1.0, .5
        # D has direct relationship to Z, Z has no relationship to M,
        # X has direct relationship to M, X has no direct relationship to Y
        d, g = 0.0, 0.0
        full_rank = False
        while not full_rank:
            E = np.random.normal(0, 2, (pm, pz))
            F = np.random.normal(0, 2, (pm, px))
            if (np.linalg.matrix_rank(E, tol=0.5) == pm) and (np.linalg.matrix_rank(F, tol=0.5) == pm):
                full_rank = True

        res = np.array(Parallel(n_jobs=-1, verbose=3)(delayed(exp_summary_multidim)(it, n, pm, pw, pz, px,
                                                                                    a, b, c, d, E, F, g)
                                                      for it in range(100)))
        pval, dval = map(np.array, zip(*res))
        print(np.mean(dval > chi2(df=px).ppf(.95)))
        print(np.mean(pval > chi2(df=pz + 1).ppf(.95)))
        assert np.isclose(np.mean(dval > chi2(df=px).ppf(.95)), 0.05, atol=2e-2)
        assert np.isclose(np.mean(pval > chi2(df=pz + 1).ppf(.95)), 0.05, atol=3e-2)


def exp_summary_strength(it, n, pw, pz, px, a, b, c, d, e, f, g, sm):
    np.random.seed(it)
    _, D, _, Z, X, Y = gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g, sm=sm)
    est = ProximalDE(dual_type='Z', cv=3, semi=True,
                     multitask=False, n_jobs=1, random_state=3, verbose=0)
    est.fit(None, D, Z, X, Y)
    id, _, _, idcrit = est.idstrength_violation_test(c=0.0)
    return id, idcrit


def test_mediator_strength_nominal_failure_prob():
    np.random.seed(123)
    pw = 1
    for n, pz, px in [(10000, 20, 10), (10000, 40, 20)]:
        # Indirect effect is a*b, direct effect is c
        a, b, c = 1.0, 1.0, .5
        # D has direct relationship to Z, Z has no relationship to M,
        # X has direct relationship to M, X has no direct relationship to Y
        d, e, f, g = 0.0, 1.0, 1.0, 0.0
        sm = 0.0
        res = np.array(Parallel(n_jobs=-1, verbose=3)(delayed(exp_summary_strength)(it, n, pw, pz, px,
                                                                                    a, b, c, d, e, f, g, sm)
                                                      for it in range(500)))
        id, idcrit = map(np.array, zip(*res))
        print(np.mean(id > idcrit), np.mean(id), np.mean(idcrit))
        assert np.isclose(np.mean(id > idcrit), 0.05, atol=3e-2)
