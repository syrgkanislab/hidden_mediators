import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from .utilities import gen_iv_data
from ..ivreg import Regularized2SLS


def test_functional_equivalency():
    ''' Test that the `Regularized2SLS` functionality for different
    input variants, mirrors exactly what we expet it to be.
    '''
    np.random.seed(123)
    n, pz, px, pw = 100, 3, 3, 0
    Z, X, Y, _ = gen_iv_data(n, pz, px, pw, .5)

    ivreg = Regularized2SLS(modelcv_first=LinearRegression(),
                            model_first=LinearRegression(),
                            model_final=LinearRegression(),
                            semi=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y)
    coef1 = ivreg.coef_
    assert np.allclose(coef1, ivreg.coef_alt_)
    tsls = LinearRegression().fit(LinearRegression().fit(Z, X).predict(Z), Y)
    coef2 = tsls.coef_
    assert np.allclose(coef1, coef2)
    assert np.allclose(ivreg.intercept_, tsls.intercept_)

    ivreg = Regularized2SLS(modelcv_first=LinearRegression(),
                            model_first=LinearRegression(),
                            model_final=LinearRegression(fit_intercept=False),
                            semi=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y)
    coef1 = ivreg.coef_
    assert np.allclose(coef1, ivreg.coef_alt_)
    coef2 = LinearRegression(fit_intercept=False).fit(LinearRegression().fit(Z, X).predict(Z), Y).coef_
    assert np.allclose(coef1, coef2)
    assert np.allclose(ivreg.intercept_, 0)

    with pytest.raises(AttributeError) as e_info:
        Regularized2SLS(modelcv_first=LinearRegression(),
                        model_first=LinearRegression(),
                        model_final=RandomForestRegressor(),
                        semi=False,
                        cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y)
    assert str(e_info.value) == ('The object `model_final` needs to have attribtes `coef_` '
                                 'and `intercept_` after being fitted and input parameter '
                                 '`fit_intercept` denoting whether an intercept was fitted.')

    coef1 = Regularized2SLS(modelcv_first=LinearRegression(fit_intercept=False),
                            model_first=LinearRegression(fit_intercept=False),
                            model_final=LinearRegression(fit_intercept=False),
                            semi=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y).coef_
    Xhat = LinearRegression(fit_intercept=False).fit(Z, X).predict(Z)
    coef2 = LinearRegression(fit_intercept=False).fit(Xhat, Y).coef_
    assert np.allclose(coef1, coef2)

    coef1 = Regularized2SLS(modelcv_first=RidgeCV(),
                            model_first=None,
                            model_final=LinearRegression(),
                            semi=False,
                            multitask=True,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y).coef_
    coef2 = LinearRegression().fit(RidgeCV().fit(Z, X).predict(Z), Y).coef_
    assert np.allclose(coef1, coef2)

    coef1 = Regularized2SLS(modelcv_first=RidgeCV(),
                            model_first=Ridge(),
                            model_final=RidgeCV(),
                            semi=True,
                            multitask=True,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y).coef_
    coef2 = RidgeCV().fit(RidgeCV().fit(Z, X).predict(Z), Y).coef_
    assert np.allclose(coef1, coef2)

    coef1 = Regularized2SLS(modelcv_first=RidgeCV(),
                            model_first=Ridge(),
                            model_final=RidgeCV(),
                            semi=False,
                            multitask=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y).coef_
    Xhat = np.hstack([RidgeCV().fit(Z, X[:, i]).predict(Z).reshape(-1, 1) for i in range(X.shape[1])])
    coef2 = RidgeCV().fit(Xhat, Y).coef_
    assert np.allclose(coef1, coef2)


def test_accuracy():
    ''' Test that `Regularized2SLS` recovers the correct coefficient
    accurately, in large enough samples.
    '''
    np.random.seed(123)
    n, px, pz, pw = 100000, 3, 3, 0
    Z, X, Y, _ = gen_iv_data(n, pz, px, pw, .7)

    coef1 = Regularized2SLS(modelcv_first=LinearRegression(),
                            model_first=LinearRegression(),
                            model_final=LinearRegression(),
                            semi=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y).coef_
    assert np.allclose(coef1, np.ones(px) / px, atol=1e-2)

    nosplitcv = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]
    for semi in [True, False]:
        for multitask in [True, False]:
            for cv in [nosplitcv, 5]:
                coef1 = Regularized2SLS(modelcv_first=RidgeCV(),
                                        model_first=Ridge(),
                                        model_final=RidgeCV(),
                                        semi=semi,
                                        multitask=multitask,
                                        cv=cv).fit(Z, X, Y).coef_
                assert np.allclose(coef1, np.ones(px) / px, atol=1e-2)


def test_minimum_norm():
    ''' Test that we recover the minimum norm solution when
    the instruments and treatments are co-linear
    '''
    np.random.seed(123)
    Z, X, Y, _ = gen_iv_data(10000, 1, 1, 0, .5)
    alphas = np.logspace(-3, 3, 100)
    ivreg = Regularized2SLS(modelcv_first=RidgeCV(fit_intercept=False, alphas=alphas),
                            model_first=Ridge(fit_intercept=False),
                            model_final=RidgeCV(fit_intercept=False, alphas=alphas),
                            semi=False,
                            cv=2, random_state=123)
    ivreg.fit(np.hstack([Z, Z, Z]), np.hstack([X, X, X]), Y)
    assert np.allclose(ivreg.coef_, np.ones(3)/3, atol=1e-2)
