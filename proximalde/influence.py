import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.utils import create_mpl_ax, annotate_axes
from statsmodels.compat.python import lrange, lzip
from matplotlib.colors import ListedColormap, Normalize
from scipy import stats
import pandas as pd
import scipy.stats as stats
from seaborn_qqplot import pplot
from .utilities import _check_input

def interpolate_color(color1, color2, t):
    return color1.reshape(1, -1) * (1 - t.reshape(-1, 1)) + color2.reshape(1, -1) * t.reshape(-1, 1)


def influence_plot(influence_measure, leverage, resid, df, nobs, *, labels=None,
                   size=48, alpha_resid=0.1, influence_measure_thr=0.5,
                   influence_measure_name='Influence Measure',
                   plot_alpha=0.75, npoints=None, ax=None):
    ''' Code is a variant adapted from the code in statsmodels
    https://github.com/statsmodels/statsmodels/blob/eef6ad3d5f18862877c691a2fa25a57df70c1aa5/statsmodels/graphics/regressionplots.py#L862
    '''
    fig, ax = create_mpl_ax(ax)

    if labels is None:
        labels = np.array(lrange(len(influence_measure)))

    mean_leverage = np.mean(leverage)

    if npoints is not None:
        # choose the largest n points with respect to each of
        # the measures and then take the union.
        inf_large_indices = np.argsort(np.abs(influence_measure))[-npoints:]
        lev_large_indices = np.argsort(np.abs(leverage))[-npoints:]
        res_large_indices = np.argsort(np.abs(resid))[-npoints:]
        indices = np.union1d(inf_large_indices, lev_large_indices)
        indices = np.union1d(indices, res_large_indices).astype(np.int64)
        influence_measure = influence_measure[indices]
        leverage = leverage[indices]
        resid = resid[indices]
        labels = labels[indices]

    psize = influence_measure
    # scale the variables
    # TODO (copied from statsmodels): what is the correct scaling and the assumption here?
    # we want plots to be comparable across different plots
    # so we would need to use the expected distribution of criterion probably
    old_range = np.ptp(psize)
    new_range = size**2 - 8**2

    psize = (psize - psize.min()) * new_range/old_range + 8**2

    ylabel = "Studentized Residuals"

    cutoff = stats.t.ppf(1. - alpha_resid / 2, df)
    large_resid = np.abs(resid) > cutoff
    large_leverage = leverage > 2. * (df + 1) / nobs
    large_cook = influence_measure > influence_measure_thr
    large_points = np.logical_or(large_resid, large_leverage)
    large_points = np.logical_or(large_points, large_cook)
    ax.scatter(leverage, resid, s=psize,
               c=interpolate_color(np.array([1, 1, 1]), np.array([0, 0, 1]),
                                   influence_measure / influence_measure.max()),
               alpha=plot_alpha)

    # add point labels
    ax = annotate_axes(np.where(large_points)[0], labels,
                       lzip(leverage, resid),
                       lzip(-(psize/2)**.5, (psize/2)**.5), "x-large",
                       ax)
    ax.axhline(-2, linestyle='--')
    ax.axhline(2, linestyle='--')
    ax.axvline(2 * mean_leverage, linestyle='--', c='red')
    ax.axvline(3 * mean_leverage, linestyle='--', c='red')

    # TODO: make configurable or let people do it ex-post?
    font = {"fontsize": 16, "color": "black"}
    ax.set_ylabel(ylabel, **font)
    ax.set_xlabel("Leverage", **font)
    ax.set_title("Influence Plot", **font)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, influence_measure.max()),
                              cmap=ListedColormap(interpolate_color(np.array([1, 1, 1]),
                                                                    np.array([0, 0, 1]),
                                                                    np.linspace(0, 1, 100)))),
        ax=ax
    )
    cbar.ax.set_ylabel(influence_measure_name)
    return fig


def _remove_diag(x):
    ''' Removes the diagonal of a square (n, n) matrix to get a
    (n, n - 1) matrix
    '''
    x_no_diag = np.ndarray.flatten(x)
    x_no_diag = np.delete(x_no_diag, range(0, len(x_no_diag), len(x) + 1), 0)
    x_no_diag = x_no_diag.reshape(len(x), len(x) - 1)
    return x_no_diag


def _exact_influence_tsls(X, Z, Y, SigmaZinv, point):
    ''' Exact leave-one-out influence calculation as described
    in the ivreg package:
    https://zeileis.github.io/ivreg/articles/Diagnostics-for-2SLS-Regression.html
    '''
    A = X.T @ Z @ SigmaZinv @ Z.T @ X
    Ainv = np.linalg.pinv(A)
    a = SigmaZinv @ Z.T @ Y
    xAx = np.sum((X @ Ainv) * X, axis=1, keepdims=True)
    c = np.sum((Z @ SigmaZinv) * Z, axis=1, keepdims=True)
    R = Z @ (X.T @ Z @ SigmaZinv).T
    xmrAx = np.sum(((X - R) @ Ainv) * X, axis=1, keepdims=True)
    xmrAxmr = np.sum(((X - R) @ Ainv) * (X - R), axis=1, keepdims=True)
    D = 1 - xAx + (xmrAx)**2 / (1 - c + xmrAxmr)
    denom = (1 - c + xmrAxmr) * D
    j = (xmrAx / denom) * (X - R) - X / D
    u = ((1 - xAx) / denom) * (X - R)
    u += (xmrAx / denom) * X
    g = u * ((Y - Z @ a) - (Y - R @ point)) + (u + j) * (Y - X @ point)

    return - g @ Ainv.T


class InfluenceDiagnostics:
    ''' Provides functionalities analogous to the unusual data influence diagnostics
    implemented in the ivreg R package:
    https://zeileis.github.io/ivreg/articles/Diagnostics-for-2SLS-Regression.html

    Parameters
    ----------
    add_constant: bool, optional (default=False)
        whether to add an intercept to instruments Z and endogenous
        regressors X at fit time.
    has_constant: bool, optional (default=False)
        whether the instrument and endogenous regressors contain an
        intercept column. If True, then the first column is assumed to
        be the intercept column.
    use_exact_influence: bool, optional (default=True)
        whether to use the exact leave-one-out influence of a sample
        on the parameters, in the robustness calculations, or use
        the asymptotic approximation.
    '''

    def __init__(self, *, add_constant=False, has_constant=False,
                 use_exact_influence=True):
        self.add_constant = add_constant
        self.has_constant = has_constant
        self.use_exact_influence = use_exact_influence

    def fit(self, Z, X, Y):
        ''' All variables are assumed to be mean-zero. Finds the solution
        to the IV moments:
            E[(Y - X'b) Z] = 0

        Parameters
        ----------
        Z: array (n, q) or (n,)
            instrument(s)
        X: array (n, p) or (n,)
            treatment(s)
        Y: (n, 1) or (n,)
            outcome

        Returns
        -------
        self : object
        '''
        # ensure all transformed into 2d matrices
        Z, X, Y = _check_input(Z, X, Y)
        assert Y.shape[1] == 1, "Y should be scalar!"

        if self.add_constant:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            Z = np.hstack([np.ones((Z.shape[0], 1)), Z])

        # 2SLS estimation
        SigmaZinv = np.linalg.pinv(Z.T @ Z)  # first stage covariance matrix
        B = SigmaZinv @ Z.T @ X  # first stage coefficients
        Xhat = Z @ B  # first stage projection of X on Z
        SigmaXhatInv = np.linalg.pinv(Xhat.T @ Xhat)  # covariance of projected X
        point = SigmaXhatInv @ Xhat.T @ Y  # final 2SLS coefficient

        # Note. When X and Z have the same dimension then this is equivalent
        # to the solution to the moment equations E[(Y - X'b) Z] = 0
        # i.e. equivalent to b = E[ZX'] E[ZY].
        J = Z.T @ X
        Jinv = np.linalg.pinv(J)
        point_alt = Jinv @ Z.T @ Y
        if X.shape[1] == Z.shape[1]:
            assert np.isclose(np.linalg.norm(point - point_alt, ord=np.inf), 0, atol=1e-4), \
                ("Just-identified setting but 2SLS different from moment solution. "
                 "Most probably IV is under-identified")

        ######
        # Robustness diagnostics
        ######

        epsilon = Y - X @ point  # residuals
        moment = (Y - X @ point) * Xhat  # moment vector
        PJinv = np.linalg.pinv(Xhat.T @ X)  # jacobian of moment

        # asymptotic influence function
        inf = moment @ PJinv.T
        cov = inf.T @ inf
        stderr = np.sqrt(np.diag(cov))

        # Exact influence of a data point
        exact_inf = _exact_influence_tsls(X, Z, Y, SigmaZinv, point)

        # dfbeta: impact on coefficient of each data point
        # one can also use the asymptotic influence function `inf`
        # for simplicity, for large sample sizes
        if self.use_exact_influence:
            dfbeta = exact_inf
        else:
            dfbeta = inf

        # Standard deviation of residuals (aka RMSE)
        s = np.std(epsilon, ddof=X.shape[1])

        # Standard deviation of deleted residuals (aka RMSE when deleting a point)
        # If sample size not too large, calculate the standard
        # deviation of the "deleted residuals". Otherwise, use s.
        if Z.shape[0] <= 10000:
            resmi = epsilon.T + dfbeta @ X.T
            resmi = _remove_diag(resmi)
            smi = np.std(resmi, axis=1, keepdims=True, ddof=X.shape[1])
        else:
            smi = s * np.ones((X.shape[0], 1))

        # hatvalues, proxies for leverage of each point
        # In the scalar case, h1_i = x_i^2 / E[E[X|Z]^2]
        hatvalues1 = np.sum((X @ SigmaXhatInv) * X, axis=1, keepdims=True)
        # In the scalar case, h2_i = E[X|z_i]^2 / E[E[X|Z]^2]
        # Equivalent to second stage regression leverage score in 2SLS
        hatvalues2 = np.sum((Xhat @ SigmaXhatInv) * Xhat, axis=1, keepdims=True)
        # In the scalar case, h3_i = |x_i * z_i| / |E[X * Z]|
        hatvalues3 = np.abs(np.sum((X @ Jinv) * Z, axis=1, keepdims=True))
        # In the scalar case, h4_i = z_i^2 / E[Z^2]
        # Equivalent to first stage regression leverage score in 2SLS
        hatvalues4 = np.sum((Z @ SigmaZinv) * Z, axis=1, keepdims=True)
        # A leverage proposed in the ivreg package to incorporate both
        # first stage and second stage leverage
        hatvalues_max = X.shape[1] * np.max(np.hstack([hatvalues2 / X.shape[1], hatvalues4 / Z.shape[1]]),
                                            axis=1, keepdims=True)

        # dffits
        dffits = np.sum(X * dfbeta, axis=1, keepdims=True)
        dffits /= (smi * np.sqrt(hatvalues1))

        # cook's distance
        cookd = np.sum(X * dfbeta, axis=1, keepdims=True)**2
        cookd /= (X.shape[1] * (s**2) * hatvalues1)

        # studentized residual
        rstudent = epsilon / (smi * np.sqrt(1 - hatvalues2))

        # standardized moment violation
        std_moment = moment / np.std(moment, axis=0, keepdims=True)
        std_violation = np.linalg.norm(std_moment, axis=1, ord=np.inf, keepdims=True)

        # storing attributes
        self.point_ = point.flatten()
        self.stderr_ = stderr.flatten()
        self.epsilon_ = epsilon.flatten()
        self.influence_ = inf
        self.l2influence_ = np.linalg.norm(inf, axis=1, ord=2)
        self.exact_influence_ = exact_inf
        self.l2exact_influence_ = np.linalg.norm(exact_inf, axis=1, ord=2)
        self.dfbeta_ = dfbeta
        self.dffits_ = dffits.flatten()
        self.cookd_ = cookd.flatten()
        self.smi_ = smi.flatten()
        self.hatvalues1_ = hatvalues1.flatten()
        self.hatvalues2_ = hatvalues2.flatten()
        self.hatvalues3_ = hatvalues3.flatten()
        self.hatvalues4_ = hatvalues4.flatten()
        self.hatvalues_max_ = hatvalues_max.flatten()
        self.rstudent_ = rstudent.flatten()
        self.standardized_violation_ = std_violation.flatten()
        self.df_ = X.shape[1]
        self.nobs_ = X.shape[0]
        self.dfresid_ = X.shape[0] - X.shape[1]

        return self

    def _check_is_fitted(self):
        if not hasattr(self, 'point_'):
            raise AttributeError("Object is not fitted!")

    def influence_plot(self, *, influence_measure='cook',
                       hatvalues_type=2, labels=None,
                       size=48, alpha_resid=0.1,
                       f_cook_thr=.2, l2inf_thr=10,
                       plot_alpha=0.75, npoints=None, ax=None):
        ''' Scatter plot of samples with leverage scores on x-axis and studentized
        residuals on y-axis. Size and transparency of point is controlled by the
        influence measure, as also designated in a colorbar. Horizontal lines are
        added at +/- 2, above which denotes large studentized residuals. Vertical
        lines are added at twice and three times the mean leverage score across all
        training samples. If npoints is used, only samples with either a large
        leverage score, or a large residual, or a large `influence_measure` are being
        plotted. The largeness of residuals is controlled by `alpha_resid`. The
        largeness of influence measures is controlled by either `f_cook_thr` or
        `l2inf_thr`, dependent on the type of influence measure. Leverage scores
        are considered large if they are at least `2. * (self.df_ + 1) / self.nobs_`.

        Parameters
        ----------
        influence_measure: {'cook', 'l2influence', 'l2exact_influence'}, (default='cook')
            measure to use for the size of each point, 'cook' means
            cook's distance. 'influence' means the l2 norm of the asymptotic
            influence of each point. 'exact_influence' means the l2 norm of
            the exact leave-one-out influence of each point.
        hatvalues_type: one of {1, 2, 3, 4, 5}, optional (default=2)
            Variants of leverage scores for 2SLS regression. For the scalar
            instrument/treatment case, these take the following meaning.
            1: h1_i = x_i^2 / E[E[X|Z]^2]
            2: h2_i = E[X|z_i]^2 / E[E[X|Z]^2] (second stage leverage)
            3: h3_i = x_i * z_i / E[X * Z]
            4: h4_i = z_i^2 / E[Z^2] (first stage leverage)
            5: h5_i = max{h2_i, h4_i}
        labels: None or array of (n_samples,), optional (default=None)
            labels for each sample. If None, then the row index will be used.
        size: int, optional (default=48)
            controls size of points
        alpha_resid: float in [0, 1], optional (default=.1)
            a t-student percentile threshold of 1-alpha_resid/2 will be used
            to define if a residual is considered ``large''
        f_cook_thr: float in [0, 1], optional (default=.2)
            a f-distribution percentile threshold of f_cook_thr will be used
            to determine if the cook
        l2inf_thr: float, optional (default=10)
            if l2influence or l2exact_influence are used, then the threshold
            for large influence will be set to inf_thr / nobs.
        plot_alpha: float in [0, 1], optional (default=.75)
            transparency of fill color of points in the plot
        npoints: None or int, optional (default=None)
            if not None, then only the union of largest n points with respect
            to the influence measure, leverage score and standardized residual
            are included in the plot.
        ax: figures axis handle

        Returns
        -------
        fig : figure
        '''
        self._check_is_fitted()
        if hatvalues_type == 1:
            leverage = self.hatvalues1_
        elif hatvalues_type == 2:
            leverage = self.hatvalues2_
        elif hatvalues_type == 3:
            leverage = self.hatvalues3_
        elif hatvalues_type == 4:
            leverage = self.hatvalues4_
        elif hatvalues_type == 5:
            leverage = self.hatvalues_max_
        else:
            raise AttributeError("Unknown hat values type")

        if influence_measure == 'cook':
            inf = self.cookd_
            inf_name = 'Cook\'s Distance'
            inf_thr = stats.f.ppf(f_cook_thr, self.df_, self.nobs_ - self.df_)
        elif influence_measure == 'l2influence':
            inf = self.l2influence_
            inf_name = '$\\ell_2$ Asymptotic Influence'
            inf_thr = l2inf_thr / self.nobs_
        elif influence_measure == 'l2exact_influence':
            inf = self.l2exact_influence_
            inf_name = '$\\ell_2$ Exact Influence'
            inf_thr = l2inf_thr / self.nobs_
        else:
            raise AttributeError("Unknown influence measure")

        residuals = self.rstudent_  # studentized residuals

        return influence_plot(inf, leverage, residuals,
                              self.df_, self.nobs_,
                              labels=labels,
                              size=size, alpha_resid=alpha_resid,
                              influence_measure_thr=inf_thr,
                              influence_measure_name=inf_name,
                              plot_alpha=plot_alpha, npoints=npoints, ax=ax)

    def cookd_plot(self, *, f_cook_thr=.2, ax=None):
        ''' Histogram plot of Cook's Distance values with a threshold
        vertical line for largeness.

        Parameters
        ----------
        f_cook_thr: float in [0, 1], optional (default=.2)
            a f-distribution percentile threshold of f_cook_thr will be used
            to determine if the cook
        ax: figures axis handle

        Returns
        -------
        fig : figure
        '''
        self._check_is_fitted()
        fig, ax = create_mpl_ax(ax)
        ax.hist(self.cookd_)
        threshold = stats.f.ppf(f_cook_thr, self.df_, self.nobs_ - self.df_)
        ax.axvline(threshold, c='red', linestyle='--')
        return fig

    def l2influence_plot(self, *, l2inf_thr=10, ax=None):
        ''' Histogram plot of Asymptotic Influence values with a threshold
        vertical line for largeness.

        Parameters
        ----------
        l2inf_thr: float, optional (default=10)
            if l2influence or l2exact_influence are used, then the threshold
            for large influence will be set to inf_thr / nobs.
        ax: figures axis handle

        Returns
        -------
        fig : figure
        '''
        self._check_is_fitted()
        fig, ax = create_mpl_ax(ax)
        ax.hist(self.l2influence_)
        threshold = l2inf_thr / self.nobs_
        ax.axvline(threshold, c='red', linestyle='--')
        return fig

    def l2exact_influence_plot(self, *, l2inf_thr=10, ax=None):
        ''' Histogram plot of Exact Leave-one-out Influence values
        with a threshold vertical line for largeness.

        Parameters
        ----------
        l2inf_thr: float, optional (default=10)
            if l2influence or l2exact_influence are used, then the threshold
            for large influence will be set to inf_thr / nobs.
        ax: figures axis handle

        Returns
        -------
        fig : figure
        '''
        self._check_is_fitted()
        fig, ax = create_mpl_ax(ax)
        ax.hist(self.l2exact_influence_)
        threshold = l2inf_thr / self.nobs_
        ax.axvline(threshold, c='red', linestyle='--')
        return fig

    def qqplot(self,):
        ''' Q-Q plot of the studentized residuals against theoretical student-t
        quantiles. Can be a diagnostic for outcome outliers.
        '''
        self._check_is_fitted()
        pplot(pd.DataFrame({'Studentized Residuals': self.rstudent_}),
              x='Studentized Residuals', y=stats.t, kind='qq', height=4, aspect=1.5,
              display_kws={"identity": False, "fit": True, "reg": True, "ci": 0.025})
