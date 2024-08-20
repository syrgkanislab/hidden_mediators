import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy


def weakiv_tests(Z, X, Y, *, demean=True, tau=.1, alpha=.05):
    
    nobs = Z.shape[0]

    Y = Y.reshape(-1, 1)
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    ####
    # De-mean instruments and regressors. Standardize instruments to have unit
    # covariance, by applying PCA essentially
    ####
    X = X - X.mean(axis=0)
    Z = Z - Z.mean(axis=0)
    Z = Z @ scipy.linalg.sqrtm(np.linalg.pinv(Z.T @ Z / nobs))

    # iv first stage
    fs = LinearRegression(fit_intercept=False).fit(Z, X)

    ######
    # Weak instrument diagnostics
    ######

    # F-statistic for weak instrument

    # Change in number of parameters from the model with and without instruments
    # Used in normalization in F-statistics below
    kz = Z.shape[1]
    resdf = nobs - Z.shape[1] - 1  # first stage residual degrees of freedom
    ftest_df1 = kz
    ftest_df2 = nobs - kz - 1

    # Non-homoskedasticity robust F-test of first stage regression
    # The numerator below is the difference in squared residuals if
    # we drop all instruments from first stage regression. It is also
    # equivalent to E[(X - E[X]) * (E[X|Z] - E[X])] = Cov(X, E[X|Z])
    Xhat = fs.predict(Z).reshape(X.shape)
    Fnonrobust = np.sum(X**2, axis=0) - np.sum((X - Xhat)**2, axis=0)
    Fnonrobust /= (kz * np.sum((X - Xhat)**2, axis=0) / resdf)
    pnonrobust = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        pnonrobust[i] = scipy.stats.f.sf(Fnonrobust[i], ftest_df1, ftest_df2)

    # Homoskedasticity robust F-test of first stage regression
    # and effect F-test for first stage regression
    # based on Montiel Olea and Pflueger.
    # see: https://cpflueger.github.io/carolinpflueger_repository/MOP_FINAL_May14.pdf
    Frobust = np.zeros(X.shape[1])
    probust = np.zeros(X.shape[1])
    Feff = np.zeros(X.shape[1])
    Keff = np.zeros(X.shape[1])
    for i in np.arange(X.shape[1]):
        # Calculate HC covariance for the coefficients in the non-constant
        # instruments
        Xi = X[:, [i]]
        W2 = sm.OLS(Xi, Z).fit(cov_type='HC1').cov_params() * nobs
        # we correct the covariance calculation due to the fact that
        # we fitted the intercept outside of OLS, implicitly, by demeaning the variables
        incpt_correction = (resdf + 1) / resdf
        W2 = incpt_correction * W2
        Frobust[i] = (Xi.T @ Z @ np.linalg.pinv(W2) @ Z.T @ Xi / (nobs * kz))[0, 0]
        probust[i] = scipy.stats.f.sf(Frobust[i], ftest_df1, ftest_df2)
        trW2 = np.trace(W2)
        Feff[i] = Xi.T @ Z @ Z.T @ Xi / (nobs * trW2)
        maxeigW2 = scipy.linalg.eigvalsh(W2)[-1]
        Keff[i] = (trW2**2) * (1 + 2 / tau) / (np.trace(W2.T @ W2) + 2 * trW2 * maxeigW2 / tau)
        Feff_crit = scipy.stats.ncx2.ppf(1 - alpha, df=Keff, nc=Keff / tau) / Keff


    return ftest_df1, ftest_df2, Fnonrobust, pnonrobust, Frobust, probust, Feff, Keff, Feff_crit