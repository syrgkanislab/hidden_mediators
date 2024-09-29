from sklearn.base import clone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from joblib import Parallel, delayed
from proximalde.gen_data import gen_data_complex, gen_data_no_controls, gen_data_with_mediator_violations, gen_data_no_controls_discrete_m
from proximalde.proximal import proximal_direct_effect, ProximalDE, residualizeW
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from proximalde.crossfit import fit_predict
from proximalde.utilities import covariance, svd_critical_value
from proximalde.proximal import residualizeW
from proximalde.proxy_rm_utils import *
from proximalde.ukbb_data_utils import *
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pk
from proximalde.ukbb_proximal import ProximalDE_UKBB


final_ss_dy = pk.load(open('final_ss_dy.pkl', 'rb'))

def run_bootstrap(D_label, Y_label, Xset, Zset, save_dir, frac=.5):
    np.random.seed(4)
    X, X_feats, Z, Z_feats = load_ukbb_XZ_data()
    Xint = get_int_feats(X_feats)
    Zint_ = get_int_feats(Z_feats)
    bad_idx = np.array([('Do not know' in x) or ('Prefer not to' in x) for x in Zint_])


    W, _, W_feats, X, X_binary, X_feats, Z, Z_binary, Z_feats, Y, D = load_ukbb_data(D_label=d, Y_label=y)
    est = ProximalDE_UKBB(binary_D=False, semi=True, cv=3, verbose=1, random_state=3)
    est.fit(W, D, Z, X, Y, D_label, Y_label, Xset=Xset, Zset=Zset, bad_idx=bad_idx)

    n = 10
    inf1 = est.bootstrap_inference(stage=1, n_subsamples=n, fraction=frac, replace=False, verbose=3, random_state=123)
    inf1.summary(save_dir=save_dir, save_fname_addn=f'_bs_stage1_nBs{n}_frac{frac}')

    # inf2 = est.bootstrap_inference(stage=2, n_subsamples=100, fraction=0.5, replace=False, verbose=3, random_state=123)
    # inf2.summary(save_dir=save_dir, save_fname_addn='_bs_stage2')

    # inf3 = est.bootstrap_inference(stage=3, n_subsamples=1000, fraction=0.5, replace=False, verbose=3, random_state=123)
    # inf3.summary(save_dir=save_dir, save_fname_addn='_bs_stage3')
    # return {'inf1': inf1, 'inf2': inf2, 'inf3':inf3}

# d,y = 'Low_inc', 'deprs'
# d,y = 'Female', 'myoc'
# d,y = 'Asian', 'OA'
# d,y = 'Black', 'chrkd'
d,y = 'On_dis', 'RA'
# d,y = 'Obese', 'OA'
point, _, _, path, (Xset, Zset) = final_ss_dy[f'{d}_{y}']
infs = run_bootstrap(d, y, Xset, Zset, path, frac=.75)
infs = run_bootstrap(d, y, Xset, Zset, path, frac=.25)
print(point)
print([(k, v.summary()) for k,v in infs.items()])