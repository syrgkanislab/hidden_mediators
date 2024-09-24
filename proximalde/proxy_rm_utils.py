from joblib import Parallel, delayed
import numpy as np
from proximalde.utilities import covariance, svd_critical_value
import scipy.stats
from tqdm import tqdm 
import os 
from proximalde.proximal import ProximalDE
import pandas as pd 

class WeakProxyRemoval():
    def __init__(self, Xres, Zres, Dres, violation_type: str = 'est', 
                 primal_type: str = 'full',  est_thresh: float = .1):
        self.Xres = Xres
        self.Zres = Zres 
        self.Dres = Dres
        assert violation_type in ['est', 'full']
        self.violation_type = violation_type
        assert primal_type in ['est', 'full']
        self.primal_type = primal_type
        self.covXD = covariance(Xres, Dres)
        self.covXZ = covariance(Xres, Zres)
        self.Zint = np.arange(Zres.shape[1])
        self.Xint = np.arange(Xres.shape[1])
        # covZ = covariance(self.Zres,self.Zres)
        # covX = covariance(self.Xres, self.Xres)

        # replacing covariance with low rank component, cleaning up the noisy eigenvalues
        U, S, Vh = scipy.linalg.svd(self.covXZ, full_matrices=False)
        Scrit = svd_critical_value(Xres, Zres)
        self.XZ_rank = (S > Scrit).sum()
        self.covXZ = U[:, S > Scrit] @ np.diag(S[S > Scrit]) @ Vh[S > Scrit, :]

        # cleaning up cov(X,D) and cov(Z,Y) to zero-out the statistical zeros
        stderr_covXD = np.sqrt(np.var((Xres - Xres.mean(axis=0)) * (Dres - Dres.mean(axis=0)), axis=0) / Xres.shape[0])
        self.covXD[np.abs(self.covXD).flatten() < 2.6 * stderr_covXD] = 0

        self.est_thresh = est_thresh
        self.dv_est_bench = self.pv_est_bench = 0 
        DZres = np.hstack([Dres, Zres])
        self.covDDZ = covariance(DZres, Dres)
        stderr_covDDZ = np.sqrt(np.var((DZres - DZres.mean(axis=0)) * (Dres - Dres.mean(axis=0)), axis=0) / DZres.shape[0])
        self.covDDZ[np.abs(self.covDDZ).flatten() < 2.6 * stderr_covDDZ] = 0
        self.covDXDZ = np.zeros((1 + Xres.shape[1], 1 + Zres.shape[1]))
        self.covDXDZ[1:, 1:] = self.covXZ
        self.covDXDZ[0, :] = self.covDDZ.flatten()
        self.covDXDZ[1:, 0] = self.covXD.flatten()

    def update_Y(self, Yres):
        self.Yres = Yres
        self.covZY = covariance(self.Zres, Yres)
        stderr_covZY = np.sqrt(np.var((self.Zres - self.Zres.mean(axis=0)) * (Yres - Yres.mean(axis=0)), axis=0) / self.Zres.shape[0])
        self.covZY[np.abs(self.covZY).flatten() < 2.6 * stderr_covZY] = 0
        self.covDZY = np.zeros((1 + self.Zres.shape[1], 1))
        self.covDZY[1:, :] = self.covZY
        self.covDZY[0, :] = covariance(self.Dres, Yres).flatten()

        self.dv_est_bench, self.pv_est_bench = self.violation_est(np.arange(self.Xres.shape[1]), np.arange(self.Zres.shape[1]))[:2]

    def change_primal_type(self, primal_type):
        assert primal_type in ['est', 'full']
        self.primal_type = primal_type
        self.dv_est_bench, self.pv_est_bench = self.violation_est(np.arange(self.Xres.shape[1]), np.arange(self.Zres.shape[1]))[:2]

    def violation_est(self, remnantX, remnantZ):
        covXZ_tmp = self.covXZ[remnantX, :][:, remnantZ]
        covZX_tmp = covXZ_tmp.T
        covXD_tmp = self.covXD[remnantX]
        covZY_tmp = self.covZY[remnantZ]
        dual_violation = np.linalg.norm(covXD_tmp - covXZ_tmp @ scipy.linalg.pinv(covXZ_tmp) @ covXD_tmp, ord=np.inf)

        if self.primal_type != 'full':
            primal_violation = np.linalg.norm(covZY_tmp - covZX_tmp @ scipy.linalg.pinv(covZX_tmp) @ covZY_tmp, ord=np.inf)
        else:
            
            ## more accurate primal violation
            covDXDZ_tmp = self.covDXDZ[[0] + [i + 1 for i in remnantX], :][:, [0] + [i + 1 for i in remnantZ]]
            covDZDX_tmp = covDXDZ_tmp.T
            covDZY_tmp = self.covDZY[[0] + [i + 1 for i in remnantZ]]
            primal_violation = np.linalg.norm(covDZY_tmp - covDZDX_tmp @ scipy.linalg.pinv(covDZDX_tmp) @ covDZY_tmp, ord=np.inf)

        return dual_violation, primal_violation, self.est_thresh * self.dv_est_bench, self.est_thresh * self.pv_est_bench 
     
    def violation_full(self, remnantX, remnantZ):
        est = ProximalDE(semi=True, cv=3, verbose=1, random_state=3)
        est.fit(None, self.Dres, self.Zres[:, remnantZ], self.Xres[:, remnantX], self.Yres)
        test_df = pd.DataFrame.from_records(est.summary().tables[2].data)
        header = test_df.iloc[0] # grab the first row for the header
        test_df = test_df[1:] # take the data less the header row
        test_df.columns = header
        primal_stat, dual_stat = test_df.statistic.iloc[1:3].to_numpy()
        primal_crit, dual_crit = test_df['critical value'].iloc[1:3].to_numpy()
        return [float(x) for x in [dual_stat, primal_stat, dual_crit, primal_crit]]
    
    def xset_trial(self, it, remnantZ, verbose, only_dual):
        ''' We try to add elements to the X's in random order, while maintaining that the dual
        violation is not violated. Here we use all the Z's, since the dual violation can only
        improve if we add more Z's.
        '''
        np.random.seed(it)
        unusedX = np.arange(self.Xres.shape[1])
        remnantX = []
        it = 0
        while len(unusedX) > 0:
            if it == 0:
                success = False
                nfirst_pair_trials = 0
                while (not success) or (nfirst_pair_trials > 10):
                    # p= self.Xres[:,remnantX[-1]] @ self.Xres[:, unusedX] / self.Xres.shape[0]
                    next = np.random.choice(len(unusedX), size=min(2, len(unusedX)), replace=False)
                    dual, _, dual_thresh, _ = self.violation_est(remnantX + unusedX[next].tolist(), remnantZ)
                    if dual < dual_thresh:
                        remnantX += unusedX[next].tolist()
                        success = True
                    nfirst_pair_trials += 1
                if success:
                    unusedX = np.delete(unusedX, next)
                else:
                    break
            else:
                next = np.random.choice(len(unusedX), size=1, replace=False)
                dual, _, dual_thresh, _ = self.violation_est(remnantX + unusedX[next].tolist(), remnantZ)
                if dual < dual_thresh:
                    remnantX += unusedX[next].tolist()
                unusedX = np.delete(unusedX, next)
            it += 1
        if remnantX:
            if self.violation_type == 'full':
                dual, primal, dual_thresh, primal_thresh = self.violation_full(remnantX, remnantZ)
                if only_dual and dual > dual_thresh:
                    return
                if not only_dual and (dual > dual_thresh or primal > primal_thresh):
                    return
            remnantX = np.sort(remnantX)
            if verbose:
                print(remnantX, self.violation_est(remnantX, remnantZ)[:2])
        
            ohe = np.zeros(self.Xres.shape[1]).astype(int)
            ohe[remnantX] = 1
            return ohe
        else:
            return None


    def zset_trial(self, it, remnantX, verbose):
        ''' Given a candidate X set, we try to add elements to the Z's in random order,
        while maintaining that the primal violation does not occur.
        '''
        np.random.seed(it)
        
        unusedZ = np.arange(self.Zres.shape[1])
        remnantZ = []
        it = 0
        while len(unusedZ) > 0:
            if it == 0:
                success = False
                nfirst_pair_trials = 0
                while (not success) or (nfirst_pair_trials > 10):
                    next = np.random.choice(len(unusedZ), size=min(2, len(unusedZ)), replace=False)
                    _, primal, _, primal_thresh = self.violation_est(remnantX, remnantZ + unusedZ[next].tolist())
                    if primal < primal_thresh:
                        remnantZ += unusedZ[next].tolist()
                        success = True
                    nfirst_pair_trials += 1
                if success:
                    unusedZ = np.delete(unusedZ, next)
                else:
                    break
            else:
                next = np.random.choice(len(unusedZ), size=1, replace=False)
                _, primal, _, primal_thresh = self.violation_est(remnantX, remnantZ + unusedZ[next].tolist())
                if primal < primal_thresh:
                    remnantZ += unusedZ[next].tolist()
                unusedZ = np.delete(unusedZ, next)
            it += 1

        if remnantZ:
            if self.violation_type == 'full':
                dual, primal, dual_thresh, primal_thresh = self.violation_full(remnantX, remnantZ)
                if dual > dual_thresh or primal > primal_thresh:
                    return
            
            remnantZ = np.sort(remnantZ)
        
            dv, pv = self.violation_est(remnantX, remnantZ)[:2]
            if verbose:
                print(remnantZ, dv, pv)
        
            ohe = np.zeros(self.Xres.shape[1] + self.Zres.shape[1]).astype(int)
            ohe[remnantX] = 1
            ohe[self.Xres.shape[1] + remnantZ] = 1
            return ohe
        else:
            return None
            # p = self.reweight_Z(unusedZ)
            # p = self.covZ[remnantZ[-1]][unusedZ]
            # p = 1/np.abs(p)
            # p /= p.sum()
            #             p = self.Zres[:,remnantZ].T @ self.Zres[:, unusedZ] / self.Zres.shape[0]
            #             p = 1 / np.abs(p).mean(axis=0)
            #             p = p/p.sum()
    def find_candidate_sets(self, ntrials, verbose=0, n_jobs=-1, 
                            save_dir='', niters=2, second_xset_thresh=2):
        unique_Zsets = np.array([np.ones(self.Zres.shape[1])]).astype(int)

        for j in range(niters):
            # we generate a set of candidate of maximal X sets such that the dual violation does not
            # occur, when we use all the Z's. Note that more Z's can only help the dual.
            candidateX = []
            for remnantZ in tqdm(unique_Zsets):
                remnantZ = np.argwhere(remnantZ).flatten()
                if second_xset_thresh == 2:
                    only_dual = (j==0)
                else:
                    only_dual = True
                candidateX += Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(self.xset_trial)(it, remnantZ, verbose, only_dual=only_dual)
                                                                for it in range(ntrials))
            candidateX = [c for c in candidateX if c is not None]
            if not candidateX:
                    return []

            candidateX = np.array(candidateX).astype(int)
            # we clean up to keep only the unique solutions
            unique_Xsets = np.unique(candidateX, axis=0)
        
            candidateXZ = []
            
            # for each unique candidate solution of X's
            for remnantX in tqdm(unique_Xsets):
                remnantX = np.argwhere(remnantX).flatten()
                # we try to construct maximal sets of Z's, such that the primal violation
                # does not occur. Note that more X's can only help the primal, which is why
                # we tried to build maximal X's in the first place.
                candidateXZ += Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(self.zset_trial)(it, remnantX, verbose)
                                                                for it in range(ntrials))

            candidateXZ = [c for c in candidateXZ if c is not None]

            if not candidateXZ:
                return []

            # this array now contains the one-hot-encodings of the Xset and the Zset (concatenated)
            candidateXZ = np.array(candidateXZ).astype(int)
            # we clean up to keep only unique Zset solutions
            unique_Zsets = np.unique(candidateXZ[:, self.Xres.shape[1]:], axis=0)

            if hasattr(self,'D_label'):
                print(self.D_label, self.Y_label)
        # we clean up to keep only unique pairs of solutions
        unique_XZsets = np.unique(candidateXZ, axis=0)
        # we transform the one hot encodings back to member sets
        final_candidates = []
        for unique_XZ in unique_XZsets:
            Xset = np.argwhere(unique_XZ[:self.Xres.shape[1]]).flatten()
            Zset = np.argwhere(unique_XZ[self.Xres.shape[1]:]).flatten()
            if self.violation_type == 'full':
                dual, primal, dual_thresh, primal_thresh = self.violation_full(remnantX, remnantZ)
            else:
                dual, primal, dual_thresh, primal_thresh = self.violation_est(remnantX, remnantZ)
            if dual < dual_thresh and primal < primal_thresh:
                final_candidates += [(Xset, Zset)]
        
        if verbose:
            print(f"Found {len(final_candidates)} candidates!")
        if save_dir:
            import pickle as pk
            pk.dump(final_candidates, open(save_dir+'candidates.pkl', 'wb'))
        return final_candidates

    def get_estimates(self, candidates, save_dir='', idx_list=[], verbose=0, npass=2, calc_rank=False):            
        if idx_list == []:
            # Compute and all 
            idx_list = np.arange(len(candidates))
        sms, points = [], []
        for idx in tqdm(idx_list):
             
            if hasattr(self,'D_label'):
                print(self.D_label, self.Y_label)
            fname = ''
            if save_dir:
                fname = save_dir + f'/{idx}/'
                if os.path.exists(fname):
                    print("Idx already calculated!")
                    continue
                else:
                    os.mkdir(fname)
            Xset, Zset = candidates[idx]
            est = ProximalDE(semi=True, cv=3, verbose=1, random_state=3)
            est.fit(None, self.Dres, self.Zres[:, Zset], self.Xres[:, Xset], self.Yres)
            sm = est.summary(save_dir=fname)

            if fname:
                est.run_diagnostics()
                inds = est.influential_set(alpha=0.05)
                np.save(f'{fname}/inf_set.npy', inds)

            df = pd.DataFrame.from_records(sm.tables[0].data)
            header = df.iloc[0] # grab the first row for the header
            df = df[1:] # take the data less the header row
            df.columns = header
            points.append(df)

            df = pd.DataFrame.from_records(sm.tables[2].data)
            header = df.iloc[0] # grab the first row for the header
            df = df[1:] # take the data less the header row
            df.columns = header
            df['pass test'] = df['pass test'].map(lambda x: x == 'True')
            sms.append(df)

            if verbose and df['pass test'].sum() > npass:
                if calc_rank:
                    svalues, svalues_crit = est.covariance_rank_test(calculate_critical=True)
                    rank_pass = np.sum(svalues >= svalues_crit)
                    print(f"Inf set size: {len(inds)} || # nonzero Cov(XZ) singular values: {rank_pass}")
                rmXset = np.setdiff1d(np.arange(self.Xres.shape[1]), Xset)
                print("Kept Xs = ", np.random.choice(self.Xint[Xset],replace=False, size=min(10,len(Xset))))
                print("Deleted Xs =", np.random.choice(self.Xint[rmXset],replace=False,size=min(10,len(rmXset))))

                rmZset = np.setdiff1d(np.arange(self.Zres.shape[1]), Zset)
                print("Kept Zs = ", np.random.choice(self.Zint[Zset],replace=False, size=min(10,len(Zset))))
                print("Deleted Zs =", np.random.choice(self.Zint[rmZset],replace=False,size=min(10,len(rmZset))))
                print(est.summary().tables[0], est.summary().tables[2])
        
        return points, sms
            