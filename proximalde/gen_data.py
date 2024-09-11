import numpy as np
import scipy.special
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from .proximal import residualizeW


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


def gen_data_no_controls_discrete_m(n, pw, pz, px, a, b, c, d, E, F, g, *, sz=1, sx=1, sy=1, pm=1):
    ''' Controls are generated but are irrelevant to the rest
    of the data. Now the mediator is multi-dimensional (takes pm
    non-zero discrete values and zero).

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
    M = np.random.binomial(1, scipy.special.expit(a * (2 * D - 1)))
    M = M.reshape(-1, 1) * np.random.multinomial(1, np.ones(pm) / pm, size=(n,))
    Z = M @ E + d * D.reshape(-1, 1) + sz * np.random.normal(0, 1, (n, pz))
    X = M @ F + sx * np.random.normal(0, 1, (n, px))
    Y = b * np.sum(M, axis=1) + c * D + g * X[:, 0] + sy * np.random.normal(0, 1, n)
    return W, D, M, Z, X, Y


def gen_data_with_mediator_violations(n, pw, pz, px, a, b, c, d, e, f, g, *, sm=2, sz=1, sx=1, sy=1):
    ''' Controls are generated but are irrelevant to the rest
    of the data. We now also have mediation paths:
        D -> Mp -> X
        Z -> Mpp -> Y
    Such paths violate the assumptions required for the method to work. The
    mediator Mp can trigger a violation of the dual test, and the mediator Mpp
    can trigger a violation of the primal test.

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
    Mp = a * D + sm * np.random.normal(0, 1, (n,))

    Z = np.zeros((n, pz))
    Z = (e * M + d * D).reshape(-1, 1) + sz * np.random.normal(0, 1, (n, pz))

    X = np.zeros((n, px))
    X[:, 0] = f * M + sx * np.random.normal(0, 1, (n))
    X[:, 1:] = f * Mp.reshape(-1, 1)

    Mpp = Z[:, 0] + sm * np.random.normal(0, 1, (n,))
    Y = b * M + b * Mpp + c * D + g * X[:, 0] + sy * np.random.normal(0, 1, n)
    return W, D, M, Z, X, Y


class SemiSyntheticGenerator:

    def __init__(self, *, split=False, test_size=.5, random_state=None):
        self.split = split
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, W, D, Z, X, Y):

        if self.split:
            train, test = train_test_split(np.arange(D.shape[0]),
                                           test_size=self.test_size,
                                           shuffle=True,
                                           random_state=self.random_state,
                                           stratify=D)
        else:
            train, test = np.arange(D.shape[0]), np.arange(D.shape[0])

        Wtrain = W[train] if W is not None else None
        _, Zres, Xres, _, *_ = residualizeW(Wtrain, D[train], Z[train], X[train], Y[train], semi=True)

        # The original covariance is U @ diag(S) @ Vh
        U, S, Vh = scipy.linalg.svd(Zres.T @ Xres / Zres.shape[0], full_matrices=False)
        # so the columnns of U are the column eigenvectors and the rows of Vh are the
        # row eigenvectors. Hence, uz = U[:, 0] and vx = Vh[0, :]
        self.uz_ = U[:, 0]
        self.vx_ = Vh[0, :]
        self.s0_ = S[0]

        if W is not None:
            if self.split:
                self.propensity_ = cross_val_predict(LogisticRegressionCV(random_state=self.random_state),
                                                     W, D,
                                                     cv=StratifiedKFold(5, shuffle=True, random_state=123),
                                                     method='predict_proba')[:, 1]
            else:
                lg = LogisticRegressionCV(random_state=self.random_state)
                self.propensity_ = lg.fit(W[train], D[train]).predict_proba(W[test])[:, 1]
        else:
            self.propensity_ = np.mean(D[train]) * np.ones(len(test))

        self.n_ = len(test)
        self.W_ = W[test] if W is not None else None
        self.D_ = D[test]
        self.Z_ = Z[test]
        self.X_ = X[test]
        self.Y_ = Y[test]
        return self

    def sample(self, nsamples, a, b, c, e, f, g, *, sm=2.0, sy=1.0, replace=True):
        if replace is False:
            assert nsamples <= self.n_, "`nsamples` should be less than train samples if replace is False"
        inds = np.random.choice(self.n_, size=nsamples, replace=replace)
        Wtilde = self.W_[inds] if self.W_ is not None else None
        Dtilde = np.random.binomial(1, self.propensity_[inds])
        Mtilde = a * Dtilde.flatten() + sm * np.random.normal(0, 1, (nsamples,))
        Ztilde = self.Z_[inds] + e * self.s0_ * Mtilde.reshape(-1, 1) * self.uz_.reshape(1, -1)
        Xtilde = self.X_[inds] + f * Mtilde.reshape(-1, 1) * self.vx_.reshape(1, -1)
        Ytilde = b * Mtilde + c * Dtilde + g * Xtilde[:, 0] + sy * np.random.normal(0, 1, (nsamples,))
        return Wtilde, Dtilde, Mtilde, Ztilde, Xtilde, Ytilde