import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.utils import create_mpl_ax, annotate_axes
from statsmodels.compat.python import lrange, lzip
from matplotlib.colors import ListedColormap, Normalize
from scipy import stats


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
