import numpy as np
import pandas as pd
from formulaic import Formula
import os

def _get_tests_dir():
    path_to_current_file = os.path.realpath(__file__)
    return os.path.dirname(path_to_current_file)

def _get_tests_data_dir():
    return os.path.join(_get_tests_dir(), "data")

def _dpath(fname):
    return os.path.join(_get_tests_data_dir(), fname)


def gen_kmenta_data():
    ''' IV data described in the ivreg R package
    diagnostics tutorial:
    https://zeileis.github.io/ivreg/articles/Diagnostics-for-2SLS-Regression.html

    Returns
    -------
    Z : array (n_samples, n_exogenous)
        instruments
    X : array (n_samples, n_endogenous)
        treatments
    Y : array (n_samples, 1)
        outcome
    labels : array (n_samples,)
        sample labels
    controls : array (n_controls,)
        the indices (of the instruments and treatments) that
        correspond to control variables, i.e. Z, X are identical
        on these indices.
    '''
    df = pd.read_csv(_dpath("Kmenta.csv"))
    Y = df['Q'].values
    X = df[['D', 'P']].values
    Z = df[['D', 'F', 'A']].values
    labels = df.iloc[:, 0]
    controls = np.array([0])
    return Z, X, Y, labels, controls

def gen_schooling_returns_data():
    df = pd.read_csv(_dpath("SchoolingReturns.csv"))
    exp2 = Formula('0 + poly(experience, 2)').get_model_matrix(df)
    exp2 = exp2.rename({'poly(experience, 2)[1]': 'experience1', 'poly(experience, 2)[2]': 'experience2'}, axis=1)
    age2 = Formula('0 + poly(age, 2)').get_model_matrix(df)
    age2 = age2.rename({'poly(age, 2)[1]': 'age1',
                        'poly(age, 2)[2]': 'age2'}, axis=1)
    dm = pd.get_dummies(df[['ethnicity', 'smsa', 'south', 'nearcollege']])
    dm.drop(['ethnicity_other', 'smsa_no', 'south_no', 'nearcollege_no'], axis=1)
    df = pd.concat([df, dm, exp2, age2], axis=1)
    Y = np.log(df['wage'].values)
    Xdf = df[['ethnicity_afam', 'smsa_yes', 'south_yes', 'education', 'experience1', 'experience2']]
    X = Xdf.values.astype(np.float64)
    Zdf = df[['ethnicity_afam', 'smsa_yes', 'south_yes', 'nearcollege_yes', 'age1', 'age2']]
    Z = Zdf.values.astype(np.float64)
    controls = np.array([0, 1, 2])
    return Z, X, Y, controls

def gen_iv_data(n, pz, px, pw, ivstrength):
    if pz < px:
        raise AttributeError("More instruments than treatments needed")

    U = np.random.normal(0, 1, size=(n, 1))

    if pw == 0:
        W = np.zeros((n, 1))
        betawx = np.ones((1, px))
        betawz = np.ones((1, pz))
        betawy = np.ones((1, 1))
    else:
        W = np.random.normal(0, 1, size=(n, pw))
        betawx = np.ones((pw, px)) / pw
        betawz = np.ones((pw, pz)) / pw
        betawy = np.ones((pw, 1)) / pw

    Z = W @ betawz + np.random.normal(0, 1, size=(n, pz))
    gamma = np.zeros((pz, px))
    gamma[:px, :px] = np.eye(px)
    X = W @ betawx + ivstrength * Z @ gamma + (1 - ivstrength) * U
    beta = np.ones((px, 1)) / px
    Y = X @ beta + W @ betawy + U

    if pw == 0:
        return Z, X, Y, None

    return np.hstack([W, Z]), np.hstack([W, X]), Y, np.arange(pw)

