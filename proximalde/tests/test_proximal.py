from ..proximal import *


def test_determinism_in_residualize_w():
    ''' Test that when random state is passed, the results
    of residualizeW are deterministic.
    '''
    return

def test_estimate_nuisances():
    ''' Test that the nuisance parameter function accurately
    recovers the desired coefficients. We pass Z, X, D, Y.
    We create two artificial data generating processes.
    In the first case, we check that eta, point_pre, correspond
    to the IV regression of Y on X, D with instrument Z, D.
    In the second case, we generate IV data, Z, X, Y and we
    pass as X, Z, Y, Y. We check that gamma, correctly recovers
    the coefficient of Y on X, using Z as an instrument.
    '''
    return

def test_estimate_final():
    ''' Test that the nuisance parameter function accurately
    recovers the desired coefficients. We pass Z, X, D, Y,
    gamma and eta. We verify that the result is the same as
    2SLS regression of Y - eta'X on D using D - gamma'Z as
    the instrument. Verify that the idstrength is the same as
    sqrt{n} * mean(D * (D - gamma'Z)) / std(D * (D - gamma'Z))
    '''
    return

def test_proximal_de_equivalency():
    ''' Verify that the `proximal_direct_effect` function gives
    the same results as the `ProximalDE` class.
    '''
    return

def test_raise_nonfitted():
    ''' Verify that the `ProximalDE` class correctly raises a
    non-fitted error when methods are called before `fit`.
    '''
    return


