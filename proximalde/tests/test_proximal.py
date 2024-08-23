import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, MultiTaskLasso
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from ..gen_data import gen_data_complex
from ..proximal import residualizeW, estimate_nuisances, estimate_final, second_stage
from .utilities import gen_iv_data
from ..ivreg import Regularized2SLS


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

    with pytest.raises(AttributeError) as e_info:
        residualizeW(Wfake, np.hstack([D.reshape(-1, 1), D.reshape(-1, 1)]),
                     Z.flatten(), X.flatten(), Y.flatten(),
                     categorical=True,
                     cv=5, semi=False, multitask=False, n_jobs=-1, verbose=0,
                     random_state=None)
    print(e_info)

    with pytest.raises(AttributeError) as e_info:
        residualizeW(Wfake, D, Z.flatten(), X.flatten(),
                     np.hstack([Y.reshape(-1, 1), Y.reshape(-1, 1)]),
                     categorical=True,
                     cv=5, semi=False, multitask=False, n_jobs=-1, verbose=0,
                     random_state=None)
    print(e_info)

    with pytest.raises(AttributeError) as e_info:
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
    Z, X, Y, _ = gen_iv_data(n, pz, px, pw, .5)

    for dual_type in ['Z', 'Q']:
        _, Ybar, eta, gamma, point_pre, std_pre, \
            primal_violation, dual_violation = estimate_nuisances(Z[:, [0]], Z[:, 1:], X[:, 1:], Y,
                                                                  dual_type=dual_type,
                                                                  cv=2, n_jobs=-1, verbose=0,
                                                                  random_state=123)
        assert np.allclose(Ybar, Y - X[:, 1:] @ eta)
        alphas = np.logspace(-3, 3, 100)
        ivreg = Regularized2SLS(modelcv_first=RidgeCV(fit_intercept=False, alphas=alphas),
                                model_first=Ridge(fit_intercept=False),
                                model_final=RidgeCV(fit_intercept=False, alphas=alphas),
                                cv=2, random_state=123).fit(Z, X, Y)
        coef1 = ivreg.coef_
        assert np.allclose(eta.flatten(), coef1[1:])
        assert np.allclose(point_pre, coef1[0])
        assert np.allclose(std_pre, ivreg.stderr_[0])
        moment = (Y - X @ coef1.reshape(-1, 1)) * Z
        pv = np.mean(moment, axis=0)
        pv = pv * np.sqrt(Z.shape[0]) / np.std(moment, axis=0)
        pv = np.max(np.abs(pv))
        assert np.allclose(pv, primal_violation)
        assert np.allclose(point_pre, 1.0, atol=5e-2)
        assert np.allclose(eta.flatten(), np.ones(px) / px, atol=5e-2)

    np.random.seed(123)
    n, pz, px, pw = 10000, 3, 3, 0
    Z, X, Y, _ = gen_iv_data(n, pz, px, pw, .5)

    Dbar, _, eta, gamma, point_pre, std_pre, \
        primal_violation, dual_violation = estimate_nuisances(Y, X, Z, Y,
                                                              dual_type='Z',
                                                              cv=2, n_jobs=-1, verbose=0,
                                                              random_state=123)
    assert np.allclose(Dbar, Y - X @ gamma)
    alphas = np.logspace(-3, 3, 100)
    ivreg = Regularized2SLS(modelcv_first=RidgeCV(fit_intercept=False, alphas=alphas),
                            model_first=Ridge(fit_intercept=False),
                            model_final=RidgeCV(fit_intercept=False, alphas=alphas),
                            cv=2, random_state=123).fit(Z, X, Y)
    coef1 = ivreg.coef_
    assert gamma.shape == (px, 1)
    assert np.allclose(gamma.flatten(), coef1)
    moment = (Y - X @ coef1.reshape(-1, 1)) * Z
    pv = np.mean(moment, axis=0)
    pv = pv * np.sqrt(Z.shape[0]) / np.std(moment, axis=0)
    pv = np.max(np.abs(pv))
    assert np.allclose(pv, dual_violation)
    assert np.allclose(gamma.flatten(), np.ones(px) / px, atol=5e-2)

    np.random.seed(123)
    n, pz, px, pw = 10000, 2, 2, 0
    Z, X, Y, _ = gen_iv_data(n, pz, px, pw, 1.0)

    Dbar, Ybar, eta, gamma, point_pre, std_pre, \
        primal_violation, dual_violation = estimate_nuisances(Y, X, X, Y,
                                                              dual_type='Q',
                                                              cv=5, n_jobs=-1, verbose=0,
                                                              random_state=123)
    # assert np.allclose(Dbar, Y - X @ gamma, atol=1e-3)
    alphas = np.logspace(-3, 3, 100)
    ivreg = Regularized2SLS(modelcv_first=RidgeCV(fit_intercept=False, alphas=alphas),
                            model_first=Ridge(fit_intercept=False),
                            model_final=RidgeCV(fit_intercept=False, alphas=alphas),
                            cv=5, random_state=123).fit(X, X, Y)
    coef1 = ivreg.coef_
    assert gamma.shape == (px, 1)
    assert np.allclose(gamma.flatten(), coef1, atol=1e-3)
    assert np.allclose(gamma.flatten(), np.ones(px) / px, atol=5e-2)

    with pytest.raises(AttributeError) as e_info:
        estimate_nuisances(Y[:100], X[:100], X[:100], Y[:100], dual_type='341324')
    assert e_info.typename == 'AttributeError'
    assert str(e_info.value) == "Unknown `dual_type`. Should be one of {'Q', 'Z'}"

    with pytest.raises(AttributeError) as e_info:
        estimate_nuisances(np.hstack([Y, Y]), X, X, Y)
    print(e_info)

    with pytest.raises(AttributeError) as e_info:
        estimate_nuisances(Y, X, X, np.hstack([Y, Y]))
    print(e_info)


def test_estimate_final():
    ''' Test that the nuisance parameter function accurately
    recovers the desired coefficients. We pass Z, X, Y.
    We verify that the result is the same as 2SLS regression of
    Y on X using Z as the instrument. Verify that the idstrength
    is the same as sqrt{n} * mean(X * Z) / std(X * Z)
    '''
    np.random.seed(123)
    Z, X, Y, _ = gen_iv_data(10000, 1, 1, 0, .5)
    point, std, ids, inf = estimate_final(Z, X, Y)
    ivreg = Regularized2SLS(modelcv_first=LinearRegression(fit_intercept=False),
                            model_first=None,
                            model_final=LinearRegression(fit_intercept=False),
                            semi=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))],
                            random_state=123).fit(Z, X, Y)
    assert np.allclose(point, ivreg.coef_[0])
    assert np.allclose(std, ivreg.stderr_[0])
    assert np.allclose(ids, np.sqrt(X.shape[0]) * np.mean(X * Z) / np.std(X * Z))
    assert np.allclose(((Y - X * point) * Z / np.mean(X * Z)).flatten(), inf)
    assert np.allclose(np.std(inf) / np.sqrt(Z.shape[0]), std)
    assert np.allclose(point, 1, atol=2e-2)
    assert np.allclose(ids * np.std(X * Z) / np.sqrt(X.shape[0]), 0.5, atol=1e-2)

    with pytest.raises(AttributeError) as e_info:
        estimate_final(np.hstack([Z, Z]), X, Y)
    print(e_info)

    with pytest.raises(AttributeError) as e_info:
        estimate_final(Z, np.hstack([X, X]), Y)
    print(e_info)

    with pytest.raises(AttributeError) as e_info:
        estimate_final(Z, X, np.hstack([Y, Y]))
    print(e_info)


def test_proximal_de_equivalency():
    ''' Verify that the `proximal_direct_effect` function gives
    the same results as the `ProximalDE` class.
    '''
    Z, X, Y, controls = gen_iv_data(10000, 3, 3, 1, .5)
    point, std, *_ = second_stage(Z[:, [0]], Z[:, 1:], X[:, 1:], Y, dual_type='Z',
                                  cv=5, n_jobs=-1, verbose=0, random_state=None)

    ivreg = Regularized2SLS(modelcv_first=LinearRegression(fit_intercept=False),
                            model_first=None,
                            model_final=LinearRegression(fit_intercept=False),
                            semi=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))],
                            random_state=123).fit(Z, X, Y)
    assert np.allclose(point, ivreg.coef_[0])
    assert np.allclose(std, ivreg.stderr_[0], atol=1e-3)


def test_raise_nonfitted():
    ''' Verify that the `ProximalDE` class correctly raises a
    non-fitted error when methods are called before `fit`.
    '''
    return


def test_pde_fit_attributes():
    return


def test_pde_conf_int():
    return


def test_pde_robust_conf_int():
    return


def test_pde_summary():
    return


def test_pde_run_diagnostics():
    return


def test_pde_subsample_bootstrap():
    return
