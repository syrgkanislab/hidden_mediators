import numpy as np
import pandas as pd
from .utilities import gen_kmenta_data, gen_schooling_returns_data, gen_iv_data
from ..ivtests import weakiv_tests

def test_ivreg_r_compatibility_kmenta():
    ''' Test that on the small KMenta dataset the results that we get
    for the non-robust F-test are the same as the results that are returned
    by the ivreg package in R:
    https://zeileis.github.io/ivreg/articles/Diagnostics-for-2SLS-Regression.html
    The numbers here have been generated by running the R package.
    '''
    Z, X, Y, _, controls = gen_kmenta_data()
    df1, df2, Fnr, pnr, _, _, _, _, _ = weakiv_tests(Z, X, Y, controls=controls)

    assert df1 == 2
    assert df2 == 16
    assert np.allclose(Fnr, [88.025])
    assert np.allclose(pnr, [2.32e-09], atol=1e-10)

    Z[0, 0] = 87.5
    df1, df2, Fnr, pnr, _, _, _, _, _ = weakiv_tests(Z, X, Y, controls=None)

    assert df1 == 3
    assert df2 == 16
    assert np.format_float_scientific(Fnr[0], precision=3) == '1.759e+06'
    assert pnr[0] < 2e-16
    assert np.format_float_scientific(Fnr[1], precision=3) == '8.917e+01'
    assert np.format_float_scientific(pnr[1], precision=3) == '3.349e-10'

    
    df1, df2, Fnr, pnr, _, _, _, _, _ = weakiv_tests(Z[:, 1:], X[:, 1:], Y, controls=None)

    assert df1 == 2
    assert df2 == 17
    assert np.round(Fnr[0], decimals=3) == 0.327
    assert np.round(pnr[0], decimals=3) == 0.725

    df1, df2, Fnr, pnr, _, _, _, _, _ = weakiv_tests(Z[:, 1], X[:, 1:].flatten(), Y.flatten())

    assert df1 == 1
    assert df2 == 18
    assert np.round(Fnr[0], decimals=3) == 0.634
    assert np.round(pnr[0], decimals=6) == 0.436338


def test_ivreg_r_compatibility_schooling_returns():
    ''' Test that on the small KMenta dataset the results that we get
    for the non-robust F-test are the same as the results that are returned
    by the ivreg package in R for the schooling returns dataset. See:
    https://zeileis.github.io/ivreg/articles/ivreg.html
    The numbers here have been generated by running the R package.
    '''
    Z, X, Y, controls = gen_schooling_returns_data()
    df1, df2, Fnr, pnr, _, _, _, _, _ = weakiv_tests(Z, X, Y, controls=controls)

    assert df1 == 3
    assert df2 == 3003
    assert np.allclose(Fnr, [8.008, 1612.707, 174.166], atol=1e-3)
    assert np.format_float_scientific(pnr[0], precision=2) == '2.58e-05'
    assert pnr[1] < 2e-16
    assert pnr[2] < 2e-16


def test_homoskedastic():
    ''' Under homoskedastic data, the three tests should yield roughly
    the same statistic for large enough samples.
    '''

    np.random.seed(123)
    for (px, pz) in [(1, 1), (1, 3), (3, 3)]:
        for pw in [0, 3]:
            Z, X, Y, controls = gen_iv_data(100000, pz, px, pw, .1)
            results = weakiv_tests(Z, X, Y, controls=controls)
            if px == pz and px == 1:
                assert np.allclose(results[2], results[4], atol=20)
                assert np.allclose(results[4], results[6], atol=1e-3)
            else:
                assert np.allclose(results[2], results[4], atol=7)
                assert np.allclose(results[4], results[6], atol=7)


def test_separation():
    ''' Test that the tests can really separate between weak instruments
    and strong instruments
    '''

    np.random.seed(123)
    for (px, pz) in [(1, 1), (1, 3), (3, 3)]:
        for pw in [0, 3]:
            Z, X, Y, controls = gen_iv_data(10000, pz, px, pw, .02)
            results = weakiv_tests(Z, X, Y, controls=controls)
            assert np.all(np.less(results[2], 10))
            assert np.all(np.less(results[4], 10))
            assert np.all(np.less(results[6], results[8]))
            Z, X, Y, controls = gen_iv_data(10000, pz, px, pw, .2)
            results = weakiv_tests(Z, X, Y, controls=controls)
            assert np.all(np.greater(results[2], 10))
            assert np.all(np.greater(results[4], 10))
            assert np.all(np.greater(results[6], results[8]))
