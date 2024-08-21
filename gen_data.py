import numpy as np
import scipy.special

def gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g):
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
    M = a * D + W[:, 0] + np.random.normal(0, 2, n)
    Z = (e * M + d * D + W[:, 0]).reshape(-1, 1) + np.random.normal(0, 1, (n, pz))
    X = (f * M + W[:, 0]).reshape(-1, 1) + np.random.normal(0, 1, (n, px))
    Y = b * M + c * D + g * X[:, 0] + W[:, 0] + np.random.normal(0, 1, n)
    return W, D, M, Z, X, Y
