import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, MultiTaskLasso
from ..gen_data import gen_data_complex
from ..proximal import residualizeW


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
    return


def test_estimate_final():
    ''' Test that the nuisance parameter function accurately
    recovers the desired coefficients. We pass Z, X, D, Y,
    gamma and eta. We verify that the result is the same as
    2SLS regression of Y - eta'X on D using D - gamma'Z as
    the instrument. Verify that the idstrength is the same as
    sqrt{n} * mean(D * (D - gamma'Z)) / std(D * (D - gamma'Z))
    '''
    return


def test_proximal_de_equivalency():
    ''' Verify that the `proximal_direct_effect` function gives
    the same results as the `ProximalDE` class.
    '''
    return


def test_raise_nonfitted():
    ''' Verify that the `ProximalDE` class correctly raises a
    non-fitted error when methods are called before `fit`.
    '''
    return