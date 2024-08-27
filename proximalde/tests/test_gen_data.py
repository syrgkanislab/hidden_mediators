import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from ..gen_data import gen_data_complex
from .utilities import gen_iv_data


def test_gen_iv_data():
    ''' Test that `gen_iv_data` raises an error with fewer
    instruments than treatments.
    '''
    with pytest.raises(AttributeError) as e_info:
        gen_iv_data(100, 1, 3, 0, .5)
    assert str(e_info.value) == "At least as many instruments as treatments needed"


def test_gen_data_complex():
    ''' Test that the data specification and generation works
    as specified.
    '''
    a = .3
    b = .6
    c = .5  # this is the direct effect we want to estimate
    d = .7
    e = .5  # if the product of e*f is small, then we have a weak instrument
    f = .8  # if the product of e*f is small, then we have a weak instrument
    g = .9
    n = 100000
    pw = 3
    pz, px = 4, 2

    np.random.seed(123)
    W, D, M, Z, X, Y = gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g)

    assert W.shape == (n, pw)
    assert D.shape == (n,)
    assert Z.shape == (n, pz)
    assert X.shape == (n, px)
    assert Y.shape == (n,)

    a_est = LinearRegression().fit(np.hstack([D.reshape(-1, 1), W]), M).coef_[0]
    assert np.isclose(a_est, a, atol=5e-2)
    b_est = LinearRegression().fit(np.hstack([D.reshape(-1, 1), M.reshape(-1, 1), X, W]), Y).coef_[1]
    assert np.isclose(b_est, b, atol=1e-2)
    c_est = LinearRegression().fit(np.hstack([D.reshape(-1, 1), M.reshape(-1, 1), X, W]), Y).coef_[0]
    assert np.isclose(c_est, c, atol=1e-2)
    d_est = LinearRegression().fit(np.hstack([D.reshape(-1, 1), M.reshape(-1, 1), W]), Z).coef_[:, 0]
    assert np.allclose(d_est, d, atol=5e-1)
    e_est = LinearRegression().fit(np.hstack([D.reshape(-1, 1), M.reshape(-1, 1), W]), Z).coef_[:, 1]
    assert np.allclose(e_est, e, atol=1e-2)
    f_est = LinearRegression().fit(np.hstack([M.reshape(-1, 1), W]), X).coef_[:, 0]
    assert np.allclose(f_est, f, atol=1e-2)
    g_est = LinearRegression().fit(np.hstack([D.reshape(-1, 1), M.reshape(-1, 1), X, W]), Y).coef_[2]
    assert np.isclose(g_est, g, atol=1e-2)
