import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, MultiTaskLassoCV, MultiTaskLasso
from sklearn.model_selection import cross_val_predict
import pytest
from ..crossfit import fit_predict
from .utilities import gen_iv_data


def test_fit_predict():
    ''' Test that the fit_predict functionality for different
    input variants, mirrors exactly what we expet it to be.
    '''

    n, px, pz, pw = 100, 3, 3, 0
    X, Z, Y, _ = gen_iv_data(n, px, pz, pw, .5)

    cvpreds = fit_predict(X, Z, LinearRegression(), None, 5, False, True, -1, 0)
    cvpreds2 = cross_val_predict(LinearRegression(), X, Z, cv=5)
    assert cvpreds.shape == cvpreds2.shape
    assert np.allclose(cvpreds, cvpreds2)

    cvpreds = fit_predict(X, Z, MultiTaskLassoCV(random_state=123),
                          MultiTaskLasso(random_state=123), 5, True, True, -1, 0)
    alpha = MultiTaskLassoCV(random_state=123).fit(X, Z).alpha_
    cvpreds2 = cross_val_predict(MultiTaskLasso(alpha=alpha, random_state=123), X, Z, cv=5)
    assert cvpreds.shape == cvpreds2.shape
    assert np.allclose(cvpreds, cvpreds2)

    cvpreds = fit_predict(X, Z, MultiTaskLassoCV(random_state=123),
                          MultiTaskLasso(random_state=123), 5, False, True, -1, 0)
    cvpreds2 = cross_val_predict(MultiTaskLassoCV(random_state=123), X, Z, cv=5)
    assert cvpreds.shape == cvpreds2.shape
    assert np.allclose(cvpreds, cvpreds2)

    with pytest.raises(AttributeError) as e_info:
        fit_predict(X, Z, LinearRegression(), None, 5, True, True, 1, 0)
    print(e_info)

    with pytest.raises(AttributeError) as e_info:
        fit_predict(X, Z, LassoCV(), LinearRegression(), 5, True, False, 1, 0)
    print(e_info)

    cvpreds = fit_predict(X, Z, LassoCV(random_state=123), None, 5, False, False, -1, 0)
    cvpreds2 = np.zeros(Z.shape)
    for i in range(Z.shape[1]):
        cvpreds2[:, i] = cross_val_predict(LassoCV(random_state=123), X, Z[:, i], cv=5)
    assert cvpreds.shape == cvpreds2.shape
    assert np.allclose(cvpreds, cvpreds2)

    cvpreds = fit_predict(X, Z, LassoCV(random_state=123), Lasso(random_state=123), 5,
                          True, False, -1, 0)
    cvpreds2 = np.zeros(Z.shape)
    for i in range(Z.shape[1]):
        alpha = LassoCV().fit(X, Z[:, i]).alpha_
        cvpreds2[:, i] = cross_val_predict(Lasso(alpha=alpha, random_state=123), X, Z[:, i], cv=5)
    assert cvpreds.shape == cvpreds2.shape
    assert np.allclose(cvpreds, cvpreds2)

    cvpreds = fit_predict(X, Z[:, 0], LassoCV(random_state=123), Lasso(random_state=123), 5,
                          True, False, -1, 0)
    alpha = LassoCV().fit(X, Z[:, 0]).alpha_
    cvpreds2 = cross_val_predict(Lasso(alpha=alpha, random_state=123), X, Z[:, 0], cv=5)
    assert cvpreds.shape == cvpreds2.shape
    assert np.allclose(cvpreds, cvpreds2)
