# Interpretable X, Z features 
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
import os 

from proximalde.ukbb_proximal import ProximalDE_UKBB
def run_inf_rm(D_label, Y_label, inf_idxs, Xset, Zset, save_dir, save_fname_addn=''):
    np.random.seed(4)
    W, _, W_feats, X, X_binary, X_feats, Z, Z_binary, Z_feats, Y, D = load_ukbb_data(D_label=d, Y_label=y)
    Z = Z[:,~bad_idx][:,Zset]
    X = X[:,Xset]
    print(D_label, Y_label, save_fname_addn)
    est = ProximalDE_UKBB(binary_D=False, semi=True, cv=3, verbose=1, random_state=3)
    est.fit(np.delete(W, inf_idxs, axis=0), np.delete(D, inf_idxs, axis=0),
             np.delete(Z, inf_idxs, axis=0), np.delete(X, inf_idxs, axis=0),
             np.delete(Y, inf_idxs, axis=0), D_label=D_label, Y_label=Y_label, save_fname_addn=f'_infRm_{D_label}{Y_label}{save_fname_addn}') 
    print(D_label, Y_label, save_fname_addn)
    return est.summary(alpha=0.05, save_dir=save_dir,save_fname_addn=f'_infRm{save_fname_addn}')

def rmNaZ(Zres, Zint):
    bad_idx = np.array([('Do not know' in x) or ('Prefer not to' in x) for x in Zint])
    Zres = Zres[:,~bad_idx]
    Zint = Zint[~bad_idx]
    return Zres, Zint

def get_median_item(y):
    srt_idx = np.argsort(y)
    if len(y) % 2 == 0: # if even elements, get max point
        i1 = len(y)//2 - 1
        i2 = len(y)//2
        if np.abs(y[srt_idx[i1]]) > np.abs(y[srt_idx[i2]]):
            return y[srt_idx[i1]], srt_idx[i1]
        else:
            return y[srt_idx[i2]], srt_idx[i2]
    else:
        return y[srt_idx[len(y)//2]], srt_idx[len(y)//2]

ss_dy = pk.load(open('ss_dy_updated_inf.pkl', 'rb'))
# ss_dy = pk.load(open('ss_dy.pkl', 'rb'))
X, X_feats, Z, Z_feats = load_ukbb_XZ_data()
Xint = get_int_feats(X_feats)
Zint_ = get_int_feats(Z_feats)
bad_idx = np.array([('Do not know' in x) or ('Prefer not to' in x) for x in Zint_])

n_ss=0
# for dy in list(ss_dy.keys()):
# for dy in ['Female_myoc']:
# for dy in ['Low_inc_deprs']:
for dy in ['On_dis_RA']:
    point, test, inf_dict, path, (Xset, Zset) = ss_dy[dy]
    for name in ['n=200', 'switch_sign']:
        inds = inf_dict[name]  
        d, y = '_'.join(dy.split('_')[:-1]), dy.split('_')[-1]
        if os.path.exists(path + f'/table0_infRm_{name}.csv'):
            pass
        else:
            run_inf_rm(D_label=d, Y_label=y, inf_idxs=inds, Xset=Xset, Zset=Zset, save_dir=path, save_fname_addn='_'+name)

