import numpy as np
import scipy.special
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from .proximal import residualizeW
from .utilities import covariance, svd_critical_value


def gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g, *, sm=2, sz=1, sx=1, sy=1):
    """
    Generates synthetic dataset for testing, where the mediator is a
    single continuous variable and only D is binary.

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
    sm : scale of noise of M
    sz : scale of noise of Z
    sx : scale of noise of X
    sy : scale of noise of Y
    """
    W = np.random.normal(0, 1, size=(n, pw))
    D = np.random.binomial(1, scipy.special.expit(2 * W[:, 0]))
    M = a * D + sm * (W[:, 0] + np.random.normal(0, 2, n))
    Z = (e * M + d * D).reshape(-1, 1) + sz * (
        W[:, [0]] + np.random.normal(0, 1, (n, pz))
    )
    X = f * M.reshape(-1, 1) + sx * (W[:, [0]] + np.random.normal(0, 1, (n, px)))
    Y = b * M + c * D + g * X[:, 0] + sy * (W[:, 0] + np.random.normal(0, 1, n))
    return W, D, M, Z, X, Y


def gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g, *, sm=2, sz=1, sx=1, sy=1):
    """Generates synthetic dataset for testing, where the mediator is a
    single continuous variable and only D is binary. .
    Controls W are generated but are irrelevant to the rest
    of the data.

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
    sm : scale of noise of M
    sz : scale of noise of Z
    sx : scale of noise of X
    sy : scale of noise of Y
    """
    W = np.random.normal(0, 1, size=(n, pw))
    D = np.random.binomial(
        1,
        0.5
        * np.ones(
            n,
        ),
    )
    M = a * D + sm * np.random.normal(0, 1, (n,))
    Z = (e * M + d * D).reshape(-1, 1) + sz * np.random.normal(0, 1, (n, pz))
    X = f * M.reshape(-1, 1) + sx * np.random.normal(0, 1, (n, px))
    Y = b * M + c * D + g * X[:, 0] + sy * np.random.normal(0, 1, n)
    return W, D, M, Z, X, Y


def gen_data_no_controls_discrete_m(
    n, pw, pz, px, a, b, c, d, E, F, g, *, sz=1, sx=1, sy=1, pm=1
):
    """Generates synthetic dataset for testing, where D is binary
    and now the mediator is multi-dimensional (size pm) and binary.
    Controls W are generated but are irrelevant to the rest
    of the data.

    n: number of samples
    pw: dimension of controls
    pm : number of non-zero discrete values that the mediator takes
    pz: dimension of treatment proxies ("instruments")
    px: dimension of outcome proxies ("treatments")
    a : strength of D -> M edge
    b : strength of M -> Y edge
    c : strength of D -> Y edge
    d : strength of D -> Z edge
    E : strength of M -> Z edge (matrix)
    F : strength of M -> X edge (matrix_)
    g : strength of X -> Y edge
    sz : scale of noise of Z
    sx : scale of noise of X
    sy : scale of noise of Y
    """
    W = np.random.normal(0, 1, size=(n, pw))
    D = np.random.binomial(1, 0.5 * np.ones(n,))
    M = np.random.binomial(1, scipy.special.expit(a * (2 * D - 1)))
    M = M.reshape(-1, 1) * np.random.multinomial(1, np.ones(pm) / pm, size=(n,))
    Z = M @ E + d * D.reshape(-1, 1) + sz * np.random.normal(0, 1, (n, pz))
    X = M @ F + sx * np.random.normal(0, 1, (n, px))
    Y = b * np.sum(M, axis=1) + c * D + g * X[:, 0] + sy * np.random.normal(0, 1, n)
    return W, D, M, Z, X, Y


def gen_data_with_mediator_violations(n, pw, pz, px, a, b, c, d, e, f, g, *,
                                      sm=2, sz=1, sx=1, sy=1, 
                                      invalidZinds=[0], invalidXinds=[0],
                                      dx_path=True, zy_path=True):
    """Generates synthetic dataset for testing, where the mediator is a
    single continuous variable and only D is binary.
    Controls W are generated but are irrelevant to the rest
    of the data. We now also have two optional mediation paths:
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
    sm : scale of noise of M
    sz : scale of noise of Z
    sx : scale of noise of X
    sy : scale of noise of Y
    invalidZinds : list
        which Z's are problematic
    invalidXinds : list
        which X's are problematic
    dx_path : boolean, if the violating D -> Mp -> X path should be generated
    zy_path : boolean, if the violating Z -> Mpp -> Y path should be generated
    """
    W = np.random.normal(0, 1, size=(n, pw))
    D = np.random.binomial(1, 0.5 * np.ones(n,))

    M = a * D + sm * np.random.normal(0, 1, (n,))
    Z = (e * M + d * D).reshape(-1, 1) + sz * np.random.normal(0, 1, (n, pz))

    X = f * M.reshape(-1, 1) + sx * np.random.normal(0, 1, (n, px))
    Mp = a * D + sm * np.random.normal(0, 1, (n,))
    if dx_path:
        X[:, invalidXinds] += f * Mp.reshape(-1, 1)

    Y = b * M + c * D + g * X[:, 0] + sy * np.random.normal(0, 1, n)
    if zy_path:
        Mpp = np.mean(Z[:, invalidZinds], axis=1) + sm * np.random.normal(0, 1, (n,))
        Y += b * Mpp

    return W, D, M, Z, X, Y


class SemiSyntheticGenerator:
    """
    Generates semi-synthetic data based on structural linear equations
    and real input dataset. The data generation process is described under
    the .fit() method.
    """

    def __init__(self, *, split: bool = False, test_size: float = 0.5, random_state=None):
        """
        Parameters:
        - split (bool): Whether to split the data into training and testing sets
        - test_size (float): The proportion of the data to include in the test split.
        - random_state (int): Random state for reproducibility.
        """
        self.split = split
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, W, D, Z, X, Y, ZXYres=None):
        """
        Fit the data generator to the training split of the input data
            if self.split else fits to all the data.

        Parameters:
        - ZXYres (list): Precomputed residualized arrays for Z, X, Y. Defaults to None,
            in which case the residuals are computed.
        """

        # Set random seed
        np.random.seed(self.random_state)

        # Compute residualized data if not provided
        if ZXYres is None:
            _, Zres, Xres, Yres, *_ = residualizeW(W, D, Z, X, Y, semi=True)
        else:
            Zres, Xres, Yres = ZXYres

        if self.split:
            train, test = train_test_split(
                np.arange(D.shape[0]),
                test_size=self.test_size,
                shuffle=True,
                random_state=self.random_state,
                stratify=D,
            )
        else:
            train, test = np.arange(D.shape[0]), np.arange(D.shape[0])

        # Singular value decomposition of Cov(ZX).
        U, S, Vh = scipy.linalg.svd(
            covariance(Zres[train], Xres[train]), full_matrices=False
        )
        # We find the statistically no-zero singular values and corresponding sub-spaces
        critical = svd_critical_value(Zres[train], Xres[train])
        G = U[:, S > critical]
        F = Vh[S > critical, :].T

        # Now we can imagine that roughly we have the structural equation
        #   Z = G M + epsilon_Z
        #   X = F M + epsilon_X
        # where epsilon_Z and epsilon_X are independent and G and F are the
        # eigenvectors found by the SVD. Note that under this structural model:
        #   Cov(Z, X) = G E[MM'] F'
        # Hence, if E[MM'] = diagonal(s1, ..., s_K), then the covariance of
        # the Z,X generated by the above structural model is the same as the
        # covariance we calculated. Thus we can generate Z, X that match this
        # covariance, by first generating a mediator M based on a normal r.v.
        # with covariance diagonal(s1, ..., s_K) and then generate Z and X.

        # To generate Z, X, we need M, G, F, as well as the distribuiton of the
        # independent random elements epsilon_Z, epsilon_X. If we project the
        # observed Z's on the orthogonal space of G and the observed X's on
        # the orthogonal space of F, then the resulting quantities will be:
        #   projZ = (I - P_G) Z = (I - P_G) G M + (I - P_G) epsilon_Z = (I - P_G) epsilon_Z
        #   projX = (I - P_F) X = (I - P_F) F M + (I - P_F) epsilon_X = (I - P_F) epsilon_X
        # thus these projected out random elements, are independent as they
        # only depend on epsilon_Z and epsilon_X. The covariance of these
        # projected r.v.s will be zero. However, they are a projected out
        # part of the epsilon's and don't contain all the independent noise.
        # Alternatively, we can take the empirical marginal distribution of X
        # and the empirical marginal distribution of Z and add a sample from
        # that to the "correlated part". This might create more noisy samples
        # but preserving more of the row data.
        projZ = np.eye(Zres.shape[1]) - G @ scipy.linalg.pinvh(G.T @ G) @ G.T
        projX = np.eye(Xres.shape[1]) - F @ scipy.linalg.pinvh(F.T @ F) @ F.T

        self.Zepsilon_ = Zres[test] @ projZ.T
        self.Xepsilon_ = Xres[test] @ projX.T

        self.G_ = G
        self.F_ = F
        self.s_ = S[S > critical]

        if self.split:
            self.propensity_ = cross_val_predict(
                LogisticRegressionCV(random_state=self.random_state, solver="saga", n_jobs=-1), 
                W, D,
                cv=StratifiedKFold(3, shuffle=True, random_state=self.random_state),
                method="predict_proba",
            )[:, 1]
        else:
            lg = LogisticRegressionCV(random_state=self.random_state)
            self.propensity_ = lg.fit(W, D).predict_proba(W)[:, 1]

        self.n_ = len(test)
        self.W_ = W[test] if W is not None else None
        self.D_ = D[test].flatten()
        self.Z_ = Z[test]
        self.X_ = X[test]
        self.Zres_ = Zres[test]
        self.Xres_ = Xres[test]
        self.Y_ = Y[test].flatten()
        self.Yres_ = Yres[test].flatten()
        return self

    def sample(
        self, nsamples, a, b, c, g, sy=1.0, replace=True, projected_epsilon=False
    ):
        """
        Sample synthetic data from the fitted generator.

        Parameters:
        - nsamples (int): Number of samples to generate.
        - a, b, c, g (float): Coefficients for structural equation model.
        - sy (float): Noise scaling factor for Y. Defaults to 1.0.
        - replace (bool): Whether to sample with replacement. Defaults to True.
        - projected_epsilon (bool): Whether to use projected residuals for noise. Defaults to False.

        Returns:
        - tuple: Synthetic data (Wtilde, Dtilde, Mtilde, Ztilde, Xtilde, Ytilde).
        """
        if replace is False:
            assert (
                nsamples <= self.n_
            ), "`nsamples` should be less than train samples if replace is False"

        inds = np.random.choice(self.n_, size=nsamples, replace=replace)
        Wtilde = self.W_[inds] if self.W_ is not None else None
        if Wtilde is not None:
            # these are the parts of X, Z, Y that are predictable from W
            baseX = self.X_[inds] - self.Xres_[inds]
            baseZ = self.Z_[inds] - self.Zres_[inds]
            baseY = self.Y_[inds] - self.Yres_[inds]
        else:
            baseX, baseZ, baseY = 0.0, 0.0, 0.0

        Dtilde = np.random.binomial(1, self.propensity_[inds])
        pm = len(self.s_)
        Mtilde = a * Dtilde.reshape(-1, 1) + np.random.multivariate_normal(
            np.zeros(pm), np.diag(self.s_), (nsamples,)
        )

        indsZ = np.random.choice(self.n_, size=nsamples, replace=replace)
        Ztilde = (
            baseZ
            + Mtilde @ self.G_.T
            + (self.Zres_[indsZ] if not projected_epsilon else self.Zepsilon_[indsZ])
        )
        indsX = np.random.choice(self.n_, size=nsamples, replace=replace)
        Xtilde = (
            baseX
            + Mtilde @ self.F_.T
            + (self.Xres_[indsX] if not projected_epsilon else self.Xepsilon_[indsX])
        )

        indsY = np.random.choice(self.n_, size=nsamples, replace=replace)
        Ytilde = (
            baseY
            + b * Mtilde @ np.ones(pm) / pm
            + c * Dtilde
            + g * Xtilde[:, 0]
            + sy * self.Yres_[indsY]
        )
        return Wtilde, Dtilde, Mtilde, Ztilde, Xtilde, Ytilde
