import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, Ridge
from .utilities import gen_iv_data
from ..ivreg import *

def test_functional_equivalency():
    ''' Test that the `Regularized2SLS` functionality for different
    input variants, mirrors exactly what we expet it to be.
    '''
    np.random.seed(123)
    n, pz, px, pw = 100, 3, 3, 0
    Z, X, Y, _ = gen_iv_data(n, pz, px, pw, .5)

    coef1 = Regularized2SLS(modelcv=LinearRegression(),
                            model=LinearRegression(),
                            semi=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y).coef_
    coef2 = LinearRegression().fit(LinearRegression().fit(Z, X).predict(Z), Y).coef_
    assert np.allclose(coef1, coef2)
    
    coef1 = Regularized2SLS(modelcv=LinearRegression(fit_intercept=False),
                            model=LinearRegression(fit_intercept=False),
                            semi=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y).coef_
    coef2 = LinearRegression(fit_intercept=False).fit(LinearRegression(fit_intercept=False).fit(Z, X).predict(Z), Y).coef_
    assert np.allclose(coef1, coef2)
    
    coef1 = Regularized2SLS(modelcv=RidgeCV(),
                            model=None,
                            semi=False,
                            multitask=True,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y).coef_
    coef2 = RidgeCV().fit(RidgeCV().fit(Z, X).predict(Z), Y).coef_
    assert np.allclose(coef1, coef2)
    
    coef1 = Regularized2SLS(modelcv=RidgeCV(),
                            model=Ridge(),
                            semi=True,
                            multitask=True,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y).coef_
    coef2 = RidgeCV().fit(RidgeCV().fit(Z, X).predict(Z), Y).coef_
    assert np.allclose(coef1, coef2)
    
    coef1 = Regularized2SLS(modelcv=RidgeCV(),
                            model=Ridge(),
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

    coef1 = Regularized2SLS(modelcv=LinearRegression(),
                            model=LinearRegression(),
                            semi=False,
                            cv=[(np.arange(X.shape[0]), np.arange(X.shape[0]))]).fit(Z, X, Y).coef_
    assert np.allclose(coef1, np.ones(px) / px, atol=1e-2)


    nosplitcv = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]
    for semi in [True, False]:
        for multitask in [True, False]:
            for cv in [nosplitcv, 5]:
                coef1 = Regularized2SLS(modelcv=RidgeCV(),
                                        model=Ridge(),
                                        semi=semi,
                                        multitask=multitask,
                                        cv=cv).fit(Z, X, Y).coef_
                assert np.allclose(coef1, np.ones(px) / px, atol=1e-2)