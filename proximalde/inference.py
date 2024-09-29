import numpy as np
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from scipy.stats import norm
import pandas as pd

def _format(res, decimals):
    arr = np.array([[res]]) if np.isscalar(res) else res.reshape(-1, 1)
    arr = np.round(arr, decimals)
    return arr


def _format_scientific(res, decimals):
    arr = np.array([res]) if np.isscalar(res) else res.flatten()
    arr = np.array([np.format_float_scientific(r, precision=decimals) for r in arr])
    arr = arr.reshape(-1, 1)
    return arr


def pvalue(zstat):
    return norm.sf(np.abs(zstat), loc=0, scale=1) * 2


class NormalInferenceResults:

    def __init__(self, point, stderr):
        self.point = np.array(point)
        self.stderr = np.array(stderr)

    def zstat(self, *, value=0):
        return (self.point - value) / self.stderr

    def conf_int(self, *, alpha=0.05):
        return norm.ppf(alpha / 2, loc=self.point, scale=self.stderr), \
               norm.ppf(1 - alpha / 2, loc=self.point, scale=self.stderr)

    def pvalue(self, *, value=0):
        return pvalue(self.zstat(value=value))

    def summary(self, *, alpha=0.05, value=0, decimals=3):
        sm = Summary()
        lb, ub = self.conf_int(alpha=alpha)
        res = np.hstack((
            _format(self.point, decimals),
            _format(self.stderr, decimals),
            _format(self.zstat(value=value), decimals),
            _format_scientific(self.pvalue(value=value), decimals),
            _format(lb, decimals),
            _format(ub, decimals)
        ))
        headers = ['point', 'stderr', 'zstat', 'pvalue', 'ci_lower', 'ci_upper']
        if len(self.point.shape) == 0:
            index = ['param']
        else:
            index = [f"param{t}" for t in range(len(self.point))]

        sm.tables.append(SimpleTable(res, headers, index, "Parameter Summary"))
        return sm


class EmpiricalInferenceResults:

    def __init__(self, point, point_dist):
        self.point = np.array(point)
        self.point_dist = np.array(point_dist)

    @property
    def stderr(self):
        return np.std(self.point_dist, axis=0)

    def conf_int(self, *, alpha=0.05, pivot=False):
        ''' Bootstrap confidence interval

        Parameters
        ----------
        alpha : float, optional (default=0.05)
            Confidence level
        pivot : bool, optional (default=False)
            Whether to use pivot bootstrap CI or percentile CI
            TODO. Implement other types of bootstrap CI versions
            similar to the boot.ci package in R.
        '''
        lower = 100 * alpha / 2
        upper = 100 * (1 - alpha / 2)
        if pivot:
            return (2 * self.point - np.percentile(self.point_dist, upper, axis=0),
                    2 * self.point - np.percentile(self.point_dist, lower, axis=0))
        else:
            return (np.percentile(self.point_dist, lower, axis=0),
                    np.percentile(self.point_dist, upper, axis=0))

    def pvalue(self, *, value=0):
        ''' pivot based p-value of the hypothesis that `param=value`

        TODO. p-values based on the bootstrap is more complicated than
        what we did here. See e.g. here for more options:
        https://stats.stackexchange.com/questions/20701/computing-p-value-using-bootstrap-with-r
        Ideally match the boot.ci results from R.
        '''
        pvalue = 2 * np.minimum((self.point_dist - self.point + value <= self.point).mean(axis=0),
                                (self.point_dist - self.point + value >= self.point).mean(axis=0))
        # in the degenerate case where every point in the distribution
        # is equal to the value tested, return nan
        return np.where(np.all(self.point_dist == value, axis=0), np.nan, pvalue)

    def summary(self, *, alpha=0.05, pivot=False, value=0, decimals=3, save_dir='', save_fname_addn=''):
        ''' Summarize all the inference results.

        TODO. Update presentation of p-value if other variants of p-values
        are also implemented.
        '''
        sm = Summary()
        lb, ub = self.conf_int(alpha=alpha, pivot=pivot)
        res = np.hstack((
            _format(self.point, decimals),
            _format(self.stderr, decimals),
            _format(lb, decimals),
            _format(ub, decimals)
        ))
        headers = ['point', 'stderr', 'ci_lower', 'ci_upper']
        if len(self.point.shape) == 0:
            index = ['param']
        else:
            index = [f"param{t}" for t in range(len(self.point))]

        sm.tables.append(SimpleTable(res, headers, index, "Parameter Summary"))
        if save_dir != '': # TO DO: DELETE SAVE_FNAME_ADDN AFTER UKBB EXPERIMENTS RUN
            pd.DataFrame(sm.tables[0]).to_csv(save_dir + f'/table0{save_fname_addn}.csv')
        return sm
