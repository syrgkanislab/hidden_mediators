import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, LogisticRegression
from sklearn.model_selection import cross_val_predict
import pytest
from ..crossfit import fit_predict
from .utilities import gen_iv_data
from ..utilities import CVWrapper


def test_fit_predict():
    ''' Test that the fit_predict functionality for different
    input variants, mirrors exactly what we expet it to be.
    '''

    n, px, pz, pw = 100, 3, 3, 0
    X, Z, Y, _ = gen_iv_data(n, px, pz, pw, .5)

    cvpreds = fit_predict(X, Z, [False] * Z.shape[1], LinearRegression(), None, 5, False, -1, 0)
    cvpreds2 = cross_val_predict(LinearRegression(), X, Z, cv=5)
    assert cvpreds.shape == cvpreds2.shape
    assert np.allclose(cvpreds, cvpreds2)

    D = np.random.binomial(1, .5, size=(X.shape[0]))
    cvpreds = fit_predict(X, D, [True] * Z.shape[1], [None], LogisticRegression(), 5, False, -1, 0)
    cvpreds2 = cross_val_predict(LogisticRegression(), X, D, cv=5, method='predict_proba')[:, 1]
    assert cvpreds.shape == cvpreds2.shape
    assert np.allclose(cvpreds, cvpreds2)

    cvpreds = fit_predict(X, Z, [False] * Z.shape[1],
                          CVWrapper(modelcv=LassoCV(random_state=123),
                                    model=Lasso(random_state=123),
                                    params=['alpha']),
                          None,
                          5, True, -1, 0)
    cvpreds2 = np.zeros(Z.shape)
    for i in range(Z.shape[1]):
        cvpreds2[:, i] = cross_val_predict(Lasso(alpha=LassoCV(random_state=123).fit(X, Z[:, i]).alpha_,
                                                 random_state=123),
                                           X, Z[:, i], cv=5)
    assert cvpreds.shape == cvpreds2.shape
    assert np.allclose(cvpreds, cvpreds2)

    cvpreds = fit_predict(X, Z, [False] * Z.shape[1],
                          CVWrapper(modelcv=LassoCV(random_state=123),
                                    model=Lasso(random_state=123),
                                    params=['alpha']),
                          None,
                          5, False, -1, 0)
    cvpreds2 = np.zeros(Z.shape)
    for i in range(Z.shape[1]):
        cvpreds2[:, i] = cross_val_predict(LassoCV(random_state=123),
                                           X, Z[:, i], cv=5)
    assert cvpreds.shape == cvpreds2.shape
    assert np.allclose(cvpreds, cvpreds2)

    with pytest.raises(AttributeError) as e_info:
        fit_predict(X, Z, [False] * Z.shape[1], LinearRegression(), None, 5, True, 1, 0)
    print(e_info)

    with pytest.raises(AttributeError) as e_info:
        fit_predict(X, D, [True] * Z.shape[1], None, LogisticRegression(), 5, True, 1, 0)
    print(e_info)
