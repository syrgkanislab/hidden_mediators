import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy

'''
DISCLAIMER: This code has now been deprecated from the main analysis.
In the main analysis we need to apply the first stage tests to the
engineered instrument Dbar = D - gamma'Z. But gamma is also estimated
from data and hence its uncertainty needs to be taken into account
when calculating the variance of the regression coefficient of Dres
on Dbar. The tests here do not do that and are intended to simply
replicate the R package weak IV tests; which essentially assume that
the instrument was not constructed using the same data and based on
some estimation procedure.

A more tailored weak iv test has been implemented in the ProximalDE
class that fixes this error.

We leave this code here, as it contains basically a replication of
all the major F-tests for weak-IVs in Python, replicating the R functionality.
'''


def weakiv_tests(Z, X, Y, *, controls=None, tau=.1, alpha=.05):
    ''' Run weak instrument tests. These tests are based on F-tests of the
    first stage regression of the treatment on the instrument, that essentially
    test for the null hypothesis that all instrument coefficients are zero in
    this first stage regression. If control indices are provided then these
    controls are partialled out and the hypothesis is that all coefficients
    other than the controls are zero.

    Parameters
    ----------
    Z : array (n_samples, n_exogenous) or (n_samples,)
        The intruments
    X : array (n_samples, n_endogenous) or (n_samples,)
        The treatments
    Y : array (n_samples, 1) or (n_samples,)
        The outcome
    controls : list or array or None, optional (default=None)
        The subset of indices of Z and X that correspond to controls.
        Z[:, controls] and X[:, controls] should be identical.
    tau : float in [0, 1], optional (default=.1)
        The target "Nagar bias" in the final IV coefficients that is used
        in calculating the desired critical values for some of the tests.
    alpha : float in [0, 1], optional (default=.05)
        The confidence level that is used in calculating the critical values
        for some of the tests.

    Returns
    -------
    ftest_df1 : int
        The first degrees of freedom parameter in the F distribution for the
        non-robust and heteroskedasticity robust F-tests.
    ftest_df2 : int
        The second degrees of freedom parameter in the F distribution for the
        non-robust and heteroskedasticity robust F-tests.
    Fnonrobust : array (n_endogenous - len(n_controls))
        The non-heteroskedasticity robust F statistic for each of the endogenous
        variables. See e.g. https://online.stat.psu.edu/stat501/lesson/6/6.2
        for more details. This statistic is correct only if the first stage
        regression residuals are homoskedastic, which is rarely the case, but
        the statistic is frequently used. A rule of thumb critical value that
        is frequently used to gauge weak instruments is 10. See e.g.:
        https://scholar.harvard.edu/files/stock/files/testing_for_weak_instruments_in_linear_iv_regression.pdf

        The statistic corresponds to the quantity
            (SSR(c) - SSR(c+i)) / (df(c) - df(c+i)) * (SSR(c+i) / df(c+i))
        where SSR(c) is the sum of squared residuals when predicting the treatment
        using only the controls and SSR(c+i), when using both controls and instruments.
        df(c) is the residual degrees of freedom in the regression that uses only
        controls and df(c+i) the residual degrees when using controls and instruments.
        This is simply df(c) = nobs - len(c) - 1 and df(c+i) = nobs - len(c+i) - 1,
        since we always also include an intercept.
    pnonrobust : array (n_endogenous - len(n_controls))
        The p-value of the null hypothesis for each of the non-robust F-tests
    Frobust : array (n_endogenous - len(n_controls))
        A heteroskedasticity robust F-test. For more details, see Definition 1 in:
        https://joseluismontielolea.com/Montiel-OleaJBES.pdf
        These definitions assume that the instruments are mean-zero and identity
        covariance. Hence a PCA transformation is applied to the original instruments
        before calculating these statistics. It roughly corresponds to the quantity:
            pi.T @ Cov(pi)^{-1} @ pi / ninstruments
        where pi is an estimate of the first stage regression and Cov(pi) is a
        heteroskedasticity robust estimate of the asymptotic covariance matrix of
        the estimate pi, i.e. sqrt{n} (pi - pi0) -> N(0, Cov(pi)).
    probust : array (n_endogenous - len(n_controls))
        The p-value of the null hypothesis for each of the robust F-test
    Frobust : array (n_endogenous - len(n_controls))
        The effective degrees of freedom F-test. For more details, see Section 2.2 in:
        https://joseluismontielolea.com/Montiel-OleaJBES.pdf
        This statistic is advocated in many prior works as the right statistic that
        correlates better with the final estimate bias, as compared to Frobust.
        See e.g.:
        https://scholar.harvard.edu/files/stock/files/nbersi2018_methods_lectures_weakiv1-2_v4.pdf
        This statistic roughly corresponds to:
            pi.T @ pi / trace(Cov(pi))
        So instead of using the inverse of the covariance matrix, we only
        keep the diagonal.
    Keff : array (n_endogenous - len(n_controls))
        The effective degrees of freedom for the Feff statistic. The statistic
        can be compared to critical values based on a non-central chi-squared distribution
        with Keff degrees of freedom and non-centrality parameter x*Keff, with
        the critical values then divided by Keff. We use the "simplified TSLS" procedure
        described in Montiel-Olea and Pflueger, where we set x=1/tau to calculate Keff.
    Feff_crit : float
        The desired critical value for the Feff statistic. The Feff statistic
        should ideally be above this level, if we want the final estimate bias to
        be roughly below the target level, specified by `tau`, and at confidence
        level specified by `alpha`. We use the "simplified TSLS" procedure described
        in Montiel-Olea and Pflueger, where we set x=1/tau to calculate this critical
        value.
    '''

    nobs = Z.shape[0]

    Y = Y.reshape(-1, 1)
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    ####
    # De-mean instruments and regressors. Standardize instruments to have unit
    # covariance, by applying PCA essentially. If controls are passed, then
    # we also residualize the controls and only keep the endogenous treatments
    # and the instruments.
    ####
    if controls is None:
        X = X - X.mean(axis=0)
        Z = Z - Z.mean(axis=0)
    else:
        W = X[:, controls]
        X = X[:, np.setdiff1d(np.arange(X.shape[1]), controls)]
        X = X - LinearRegression().fit(W, X).predict(W).reshape(X.shape)
        Z = Z[:, np.setdiff1d(np.arange(Z.shape[1]), controls)]
        Z = Z - LinearRegression().fit(W, Z).predict(W).reshape(Z.shape)
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
    implicit_df = 1 if controls is None else 1 + len(controls)
    resdf = nobs - Z.shape[1] - implicit_df  # first stage residual degrees of freedom
    ftest_df1 = kz
    ftest_df2 = resdf

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
        incpt_correction = (nobs - Z.shape[1]) / resdf
        W2 = incpt_correction * W2
        Frobust[i] = (Xi.T @ Z @ np.linalg.pinv(W2) @ Z.T @ Xi / (nobs * kz))[0, 0]
        probust[i] = scipy.stats.f.sf(Frobust[i], ftest_df1, ftest_df2)
        trW2 = np.trace(W2)
        Feff[i] = (Xi.T @ Z @ Z.T @ Xi / (nobs * trW2))[0, 0]
        maxeigW2 = scipy.linalg.eigvalsh(W2)[-1]
        Keff[i] = (trW2**2) * (1 + 2 / tau) / (np.trace(W2.T @ W2) + 2 * trW2 * maxeigW2 / tau)
        Feff_crit = scipy.stats.ncx2.ppf(1 - alpha, df=Keff, nc=Keff / tau) / Keff

    return ftest_df1, ftest_df2, Fnonrobust, pnonrobust, Frobust, probust, Feff, Keff, Feff_crit
