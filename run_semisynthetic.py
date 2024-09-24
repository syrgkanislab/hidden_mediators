import numpy as np
from joblib import Parallel, delayed
from proximalde.ukbb_proximal import residualizeW_ukbb
from proximalde.proximal import ProximalDE
from proximalde.gen_data import SemiSyntheticGenerator
from proximalde.ukbb_data_utils import *
import argparse 

def minmax(M):
    Mmin =  M.min(axis=0,keepdims=True)
    Mmax =  M.max(axis=0,keepdims=True)
    zero_divide = (Mmin == Mmax).squeeze()
    M = (M - Mmin) / (Mmax - Mmin)
    if zero_divide.sum():
        M[:, zero_divide] = Mmin.squeeze()[zero_divide]+.5
    return M

def _make_Z_binary(Z,sample=False):
    if sample:
        Z[:,Z_binary] = minmax(Z[:,Z_binary])
        Z[:,Z_binary] = np.concatenate(Parallel(n_jobs=-1, verbose=0)(delayed(lambda v: np.random.binomial(1, v))(Z[:,i])
                                        for i in np.argwhere(Z_binary)),axis=1)
    else:
        Z[:, Z_binary] =  (Z[:, Z_binary]> 0).astype(int)
    return Z

def proximal_est(it, generator, n, a, b, c, g, *, sy=1.0,n_splits=3, binary_D=True,
                 model_classification='linear', model_regression='linear',
            n_jobs=-1, verbose=0, make_Z_binary=False, classify_Z=False, sample_Z_binary=False):
    np.random.seed(it)
    
    # M is unobserved so we omit it from the return variables
    Wt, Dt, _, Zt, Xt, Yt = generator.sample(n, a, b, c, g, sy=sy, replace=True)

    if make_Z_binary:
        Zt = _make_Z_binary(Zt,sample=sample_Z_binary)
    if classify_Z:
        binary_Z = Z_binary
    else:
        binary_Z = []
    est = ProximalDE(cv=n_splits, semi=True, binary_D=binary_D, model_regression=model_regression,
                    model_classification=model_classification, binary_Z=binary_Z,
                    n_jobs=n_jobs, random_state=it, verbose=verbose)
    est.fit(Wt, Dt, Zt, Xt, Yt)
    weakiv_stat, _, _, weakiv_crit = est.weakiv_test(alpha=0.05)
    idstr, _, _, idstr_crit = est.idstrength_violation_test(alpha=0.05)
    pval, _, _, pval_crit = est.primal_violation_test(alpha=0.05)
    dval, _, _, dval_crit = est.dual_violation_test(alpha=0.05)
    lb, ub = est.robust_conf_int(lb=-2, ub=2)
    return est.point_, est.stderr_, est.r2D_, est.r2Z_, est.r2X_, est.r2Y_, \
        idstr, idstr_crit, est.point_pre_, est.stderr_pre_, \
        pval, pval_crit, dval, dval_crit, weakiv_stat, weakiv_crit, \
        lb, ub

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run parameterized experiment with UKBB data")

    # Add command line arguments
    parser.add_argument('--use_Dreal', action='store_true')
    parser.add_argument('--make_Z_binary', action='store_true')
    parser.add_argument('--sample_Z_binary', action='store_true')
    parser.add_argument('--classify_Z', action='store_true')
    parser.add_argument('--binary_D', action='store_true')
    parser.add_argument('--model_clsf', type=str, default='xgb')
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()


    D_label = 'Obese'
    Y_label = 'back'
    W, W_binary, W_feats, X, X_binary, X_feats, Z, Z_binary, Z_feats, Y, D = load_ukbb_data(D_label=D_label, Y_label=Y_label)

    _, X_feats, _, Z_feats = load_ukbb_XZ_data()
    Xint = get_int_feats(X_feats)
    Zint = get_int_feats(Z_feats)
    Dres, Zres, Xres, Yres = residualizeW_ukbb(W, D, Z, X, Y, D_label=D_label, Y_label=Y_label, save_fname_addn='', random_state=3, cv=3)[:4] 
    bad_idx = np.array([('Do not know' in x) or ('Prefer not to' in x) for x in Zint])
    Zres = Zres[:,~bad_idx]
    Zint = Zint[~bad_idx]
    Z = Z[:, ~bad_idx]
    Z_binary = Z_binary[~bad_idx]
    print(Zres.shape)
    np.random.seed(0)
    generator = SemiSyntheticGenerator(split=True)
    generator.fit(W, D, Z, X, Y, ZXYres=[Zres, Xres, Yres],
                resample_D=not args.use_Dreal, propensity=np.load('propensity.npy'))

    nsamples = 50000
    a = 1.0  # a*b is the indirect effect through mediator
    b = 1.0
    c = .5  # this is the direct effect we want to estimate
    d = .0  # this can be zero; does not hurt
    e = 1.0  # if the product of e*f is small, then we have a weak instrument
    f = 1.0  # if the product of e*f is small, then we have a weak instrument
    g = .0  # this can be zero; does not hurt


    results = []
    i = 100
    saved = 0
    print(args)
    from tqdm import tqdm
    pbar = tqdm(total = saved)

    while saved < args.iters:
        try:
            results.append(proximal_est(i,generator,nsamples,a,b,c,g,make_Z_binary=args.make_Z_binary, 
                                        model_classification=args.model_clsf, binary_D=args.binary_D,
                                        classify_Z=args.classify_Z, sample_Z_binary=args.sample_Z_binary))
            saved += 1
            i += 1
            pbar.update(1)
        except ValueError as e:
            print(e)
            i += 1
    print(args)
    import pickle as pk
    pk.dump(results,open(f'./results/semisynthetic/results_Dreal{args.use_Dreal}_Zbinary{args.make_Z_binary}_ClsfZ{args.classify_Z}_SmpZ{args.sample_Z_binary}_Clsf{args.model_clsf}_Dbinary{args.binary_D}.pkl', 'wb'))
    print(1/0)
