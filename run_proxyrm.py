import numpy as np
from proximalde.proxy_rm_utils import WeakProxyRemoval
from proximalde.ukbb_data_utils import load_ukbb_XZ_data, get_int_feats, load_ukbb_res_data
import os
import argparse
from tqdm import tqdm
import pickle as pk 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameterized experiment with UKBB data")

    # Add command line arguments
    parser.add_argument('--est_thresh', type=float, default=0.1, help='Threshold for pthresh')
    parser.add_argument('--primal_type', type=str, default='full', help='Primal type to change to')
    parser.add_argument('--ntrials', type=int, default=100, help='Number of candidate sets to generate')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--nest', type=int, default=100, help='Number of estimates to generate')
    parser.add_argument('--D_label', type=str, required=True, help='D_label for dataset loading')
    parser.add_argument('--rm_na_Z', action='store_true', help='Remove Z feats with NA answers')
    parser.add_argument('--fem_only', action='store_true')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--violation_type', type=str, default='full', help='')

    args = parser.parse_args()
    D_label = args.D_label
    X, X_feats, Z, Z_feats = load_ukbb_XZ_data()
    Xint = get_int_feats(X_feats)
    Zint_ = get_int_feats(Z_feats)

    if not args.fem_only:
        Xres, Zres, _, Dres = load_ukbb_res_data(D_label, Y_label='OA')

        rmnaZ = ''
        if args.rm_na_Z:
            bad_idx = np.array([('Do not know' in x) or ('Prefer not to' in x) for x in Zint_])
            Zres = Zres[:,~bad_idx]
            Zint = Zint_[~bad_idx]
            rmnaZ = 'rmNaZ_'
            print(Zres.shape)
        prm = WeakProxyRemoval(Xres,Zres,Dres,est_thresh=args.est_thresh,
                            primal_type=args.primal_type, violation_type=args.violation_type)
        prm.Xint = Xint
        prm.Zint = Zint
        prm.D_label = D_label
    rerun = ['No_priv_insr_endo', 'Low_inc_infl', 'Low_inc_mela', 'No_priv_insr_copd', 'No_priv_insr_mela']
    Y_labels = ['OA', 'myoc','deprs', 'back', 'RA', 'fibro', 'infl', 'copd','chrkd','mgrn','mela']
    if args.fem_only:
        Y_labels = ['preg', 'endo']
    for Y_label in tqdm(Y_labels):
        if args.rerun:
            if f'{D_label}_{Y_label}' not in rerun:
                continue

        try:
            print(Y_label)
            if args.fem_only:
                Xres, Zres, Yres, Dres = load_ukbb_res_data(D_label, Y_label)
                rmnaZ = ''
                if args.rm_na_Z:
                    bad_idx = np.array([('Do not know' in x) or ('Prefer not to' in x) for x in Zint_])
                    Zres = Zres[:,~bad_idx]
                    Zint = Zint_[~bad_idx]
                    rmnaZ = 'rmNaZ_'
                    print(Zres.shape)
                prm = WeakProxyRemoval(Xres,Zres,Dres,est_thresh=args.est_thresh,
                                    primal_type=args.primal_type, violation_type=args.violation_type)
                prm.Xint = Xint
                prm.Zint = Zint
                prm.D_label = D_label
            else:
                Yres = load_ukbb_res_data(D_label, Y_label)[2]
            prm.update_Y(Yres)
            prm.Y_label = Y_label

            BASE_DIR = f'./results/proxyrm/{D_label}_{Y_label}/'
            if not os.path.exists(BASE_DIR):
                os.mkdir(BASE_DIR)
            save_dir = BASE_DIR + f'{rmnaZ}ntrials{args.ntrials}_eth{prm.est_thresh}_pty{prm.primal_type}_vtype={args.violation_type}_Rgrs=linear/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            

            np.random.seed(args.seed)
            if os.path.exists(save_dir+'candidates.pkl'):
                candidates = pk.load(open(save_dir+'candidates.pkl', 'rb'))
            else:
                candidates = prm.find_candidate_sets(args.ntrials, save_dir=save_dir)
            print(len(candidates))
            # Randomly select indices for get_estimates
            print((len(os.listdir(save_dir)) - 1), len(candidates))
            if (len(os.listdir(save_dir)) - 1) < len(candidates):
                np.random.seed(args.seed)
                idx_list = np.random.choice(np.arange(len(candidates)), 
                                            replace=False, size=min(len(candidates), args.nest))
                prm.get_estimates(candidates, idx_list=idx_list, save_dir=save_dir, calc_rank=False, verbose=1, npass=2)
        except ZeroDivisionError as e:
            print(e)