import numpy as np
import scipy.special


def gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g, *, sm=2, sz=1, sx=1, sy=1):
    '''
    n: number of samples
    pw: dimension of controls
    pz: dimension of treatment proxies ("instruments")
    px: dimension of outcome proxies ("treatments")
    a : strength of D -> M edge
    b : strength of M -> Y edge
    c : strength of D -> Y edge
    d : strength of D -> Z edge
    e : strength of M -> Z edge
    f : strength of M -> X edge
    g : strength of X -> Y edge
    '''
    W = np.random.normal(0, 1, size=(n, pw))
    D = np.random.binomial(1, scipy.special.expit(2 * W[:, 0]))
    M = a * D + sm * (W[:, 0] + np.random.normal(0, 2, n))
    Z = (e * M + d * D).reshape(-1, 1) + sz * (W[:, [0]] + np.random.normal(0, 1, (n, pz)))
    X = f * M.reshape(-1, 1) + sx * (W[:, [0]] + np.random.normal(0, 1, (n, px)))
    Y = b * M + c * D + g * X[:, 0] + sy * (W[:, 0] + np.random.normal(0, 1, n))
    return W, D, M, Z, X, Y


def gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g, *, sm=2, sz=1, sx=1, sy=1):
    ''' Controls are generated but are irrelevant to the rest
    of the data

    n: number of samples
    pw: dimension of controls
    pz: dimension of treatment proxies ("instruments")
    px: dimension of outcome proxies ("treatments")
    a : strength of D -> M edge
    b : strength of M -> Y edge
    c : strength of D -> Y edge
    d : strength of D -> Z edge
    e : strength of M -> Z edge
    f : strength of M -> X edge
    g : strength of X -> Y edge
    '''
    W = np.random.normal(0, 1, size=(n, pw))
    D = np.random.binomial(1, .5 * np.ones(n,))
    M = a * D + sm * np.random.normal(0, 1, (n,))
    Z = (e * M + d * D).reshape(-1, 1) + sz * np.random.normal(0, 1, (n, pz))
    X = f * M.reshape(-1, 1) + sx * np.random.normal(0, 1, (n, px))
    Y = b * M + c * D + g * X[:, 0] + sy * np.random.normal(0, 1, n)
    return W, D, M, Z, X, Y


def gen_data_no_controls_discrete_m(n, pw, pz, px, a, b, c, d, e, f, g, *, sm=2, sz=1, sx=1, sy=1, pm=1):
    ''' Controls are generated but are irrelevant to the rest
    of the data

    n: number of samples
    pw: dimension of controls
    pz: dimension of treatment proxies ("instruments")
    px: dimension of outcome proxies ("treatments")
    a : strength of D -> M edge
    b : strength of M -> Y edge
    c : strength of D -> Y edge
    d : strength of D -> Z edge
    e : strength of M -> Z edge
    f : strength of M -> X edge
    g : strength of X -> Y edge
    '''
    W = np.random.normal(0, 1, size=(n, pw))
    D = np.random.binomial(1, .5 * np.ones(n,))
    M = np.random.binomial(pm, scipy.special.expit(a * D))

    Z = d * D.reshape(-1, 1) + sz * np.random.normal(0, 1, (n, pz))
    X = sx * np.random.normal(0, 1, (n, px))
    for m in range(1, pm + 1):
        Z += (M == m).reshape(-1, 1) * e[m - 1].reshape(1, -1)
        X += (M == m).reshape(-1, 1) * f[m - 1].reshape(1, -1)

    Y = b * M + c * D + g * X[:, 0] + sy * np.random.normal(0, 1, n)
    return W, D, M, Z, X, Y
