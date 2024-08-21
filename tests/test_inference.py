import numpy as np
import statsmodels.api as sm
from .utilities import gen_iv_data
from ..inference import *

def test_normal_inference():
    ''' Test `NormalInference` functionality. Compare to statsmodels.
    Check that it can handle multi-dimensional parameters and scalar
    parameters.
    '''
    np.random.seed(123)
    n, px, pz, pw = 100, 1, 1, 0
    Z, X, _, _ = gen_iv_data(n, pz, px, pw, .5)

    res = sm.OLS(X, Z).fit(use_t=False)
    stderr = np.sqrt(res.cov_params()).flatten()
    point = res.params

    inf = NormalInferenceResults(point, stderr)
    assert np.allclose(inf.pvalue(), res.pvalues, atol=1e-20)
    assert np.allclose(inf.zstat(), res.tvalues)
    lb, ub = inf.conf_int(alpha=.1)
    ci = res.conf_int(alpha=.1)
    lb2, ub2 = ci[:, 0], ci[:, 1]
    assert np.allclose(lb, lb2)
    assert np.allclose(ub, ub2)

    dc = 10
    tbl = inf.summary(alpha=0.1, value=1, decimals=dc).tables[0]
    assert tbl[1][1].data == f'{np.round(inf.point, dc)[0]}'
    assert tbl[1][2].data == f'{np.round(inf.stderr, dc)[0]}'
    assert tbl[1][3].data == f'{np.round(inf.zstat(value=1), dc)[0]}'
    assert tbl[1][4].data == np.format_float_scientific(inf.pvalue(value=1)[0], precision=dc)
    assert tbl[1][5].data == f'{np.round(inf.conf_int(alpha=0.1)[0][0], dc)}'
    assert tbl[1][6].data == f'{np.round(inf.conf_int(alpha=0.1)[1][0], dc)}'

    inf2 = NormalInferenceResults(np.array([point[0], point[0]]), np.array([stderr[0], stderr[0]]))
    assert np.allclose(inf.pvalue(), inf2.pvalue()[1], atol=1e-20)
    assert np.allclose(inf.zstat(), inf2.zstat()[1])
    ci = inf2.conf_int(alpha=.1)
    lb2, ub2 = ci[0][1:], ci[1][1:]
    assert np.allclose(lb, lb2)
    assert np.allclose(ub, ub2)

    tbl2 = inf2.summary(alpha=0.1, value=1, decimals=dc).tables[0]
    assert tbl2[1][0].data == 'param0'
    assert tbl2[2][0].data == 'param1'
    assert tbl2[1][1].data == f'{np.round(inf.point, dc)[0]}'
    assert tbl2[2][1].data == f'{np.round(inf.point, dc)[0]}'
    assert tbl2[1][2].data == f'{np.round(inf.stderr, dc)[0]}'
    assert tbl2[2][2].data == f'{np.round(inf.stderr, dc)[0]}'

    inf3 = NormalInferenceResults(point[0], stderr[0])
    assert np.isclose(inf.pvalue()[0], inf3.pvalue(), atol=1e-20)
    assert np.isclose(inf.zstat()[0], inf3.zstat())
    ci = inf3.conf_int(alpha=.1)
    lb2, ub2 = ci[0], ci[1]
    assert np.isclose(lb[0], lb2)
    assert np.isclose(ub[0], ub2)

    tbl3 = inf3.summary(alpha=0.1, value=1, decimals=dc).tables[0]
    assert tbl3[1][0].data == 'param'
    assert tbl3[1][1].data == f'{np.round(inf.point[0], dc)}'
    assert tbl3[1][2].data == f'{np.round(inf.stderr[0], dc)}'


def test_empirical_inference():
    ''' Test that empirical inference produces comparable results
    as advertised, when data distribution is actually a Normal,
    by comparing to the `NormalInferenceResults` outputs.
    '''
    np.random.seed(123)
    stderr = np.array([1])
    point = np.array([2])
    point_dist = np.random.normal(point[0], stderr[0], size=(1000000, 1))

    inf = NormalInferenceResults(point, stderr)
    inf2 = EmpiricalInferenceResults(point, point_dist)
    assert np.allclose(inf.stderr, inf2.stderr, atol=1e-2)
    assert np.allclose(inf.conf_int(alpha=.1)[0], inf2.conf_int(alpha=.1)[0], atol=1e-2)
    assert np.allclose(inf.conf_int(alpha=.1)[1], inf2.conf_int(alpha=.1)[1], atol=1e-2)
    assert np.allclose(inf.conf_int(alpha=.1)[0], inf2.conf_int(alpha=.1, pivot=True)[0], atol=1e-2)
    assert np.allclose(inf.conf_int(alpha=.1)[1], inf2.conf_int(alpha=.1, pivot=True)[1], atol=1e-2)
    assert np.allclose(inf.pvalue(value=1), inf2.pvalue(value=1), atol=1e-2)

    tbl2 = inf2.summary(alpha=0.1, decimals=3).tables[0]
    assert tbl2[1][0].data == 'param0'
    assert tbl2[1][1].data == 2
    assert np.allclose(tbl2[1][2].data, stderr[0], atol=1e-2)
    assert np.allclose(tbl2[1][3].data, np.round(inf2.conf_int(alpha=0.1)[0], decimals=3))
    assert np.allclose(tbl2[1][4].data, np.round(inf2.conf_int(alpha=0.1)[1], decimals=3))

    
    inf = NormalInferenceResults(point[0], stderr[0])
    inf2 = EmpiricalInferenceResults(point[0], point_dist[:, 0])
    assert np.allclose(inf.stderr, inf2.stderr, atol=1e-2)
    assert np.allclose(inf.conf_int(alpha=.1)[0], inf2.conf_int(alpha=.1)[0], atol=1e-2)
    assert np.allclose(inf.conf_int(alpha=.1)[1], inf2.conf_int(alpha=.1)[1], atol=1e-2)
    assert np.allclose(inf.conf_int(alpha=.1)[0], inf2.conf_int(alpha=.1, pivot=True)[0], atol=1e-2)
    assert np.allclose(inf.conf_int(alpha=.1)[1], inf2.conf_int(alpha=.1, pivot=True)[1], atol=1e-2)
    assert np.allclose(inf.pvalue(value=1), inf2.pvalue(value=1), atol=1e-2)

    tbl2 = inf2.summary(alpha=0.1, decimals=3).tables[0]
    assert tbl2[1][0].data == 'param'
    assert tbl2[1][1].data == 2
    assert np.allclose(tbl2[1][2].data, stderr[0], atol=1e-2)
    assert np.allclose(tbl2[1][3].data, np.round(inf2.conf_int(alpha=0.1)[0], decimals=3))
    assert np.allclose(tbl2[1][4].data, np.round(inf2.conf_int(alpha=0.1)[1], decimals=3))
