from joblib import Parallel, delayed
import numpy as np
from proximalde.utilities import covariance, svd_critical_value
import scipy.stats
from tqdm import tqdm 
import os 
from proximalde.proximal import ProximalDE
import pandas as pd 

class WeakProxyRemoval():
    def __init__(self, Xres, Zres, Dres, primal_type: str = 'full', pthresh: float = .1, dthresh: float = .1):
        self.Xres = Xres
        self.Zres = Zres 
        self.Dres = Dres
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
        self.covXZ = U[:, S > Scrit] @ np.diag(S[S > Scrit]) @ Vh[S > Scrit, :]

        # cleaning up cov(X,D) and cov(Z,Y) to zero-out the statistical zeros
        stderr_covXD = np.sqrt(np.var((Xres - Xres.mean(axis=0)) * (Dres - Dres.mean(axis=0)), axis=0) / Xres.shape[0])
        self.covXD[np.abs(self.covXD).flatten() < 1.96 * stderr_covXD] = 0

        self.pthresh = pthresh
        self.dthresh = dthresh

        DZ = np.hstack([Dres, Zres])
        self.covDDZ = covariance(Dres, DZ)
        self.covDXDZ = np.zeros((1 + Xres.shape[1], 1 + Zres.shape[1]))
        self.covDXDZ[1:, 1:] = self.covXZ
        self.covDXDZ[0, :] = self.covDDZ.flatten()
        self.covDXDZ[1:, 0] = self.covXD.flatten()

    def update_Y(self, Yres):
        self.Yres = Yres
        self.covZY = covariance(self.Zres, Yres)
        stderr_covZY = np.sqrt(np.var((self.Zres - self.Zres.mean(axis=0)) * (Yres - Yres.mean(axis=0)), axis=0) / self.Zres.shape[0])
        self.covZY[np.abs(self.covZY).flatten() < 1.96 * stderr_covZY] = 0
        self.covDZY = np.zeros((1 + self.Zres.shape[1], 1))
        self.covDZY[1:, :] = self.covZY
        self.covDZY[0, :] = covariance(self.Dres, Yres).flatten()

        self.dv_bench, self.pv_bench = self.violation(np.arange(self.Xres.shape[1]), np.arange(self.Zres.shape[1]))

    def reweight_Z(self, idx, naToOtherRatio:int = 100000):
        """ UKBB specific, SHOULD DELETE after"""
        Zset = (1-np.array([('Do not know' in x) or ('Prefer not to' in x) for x in self.Zint[idx]]).astype(int))
        Zset = Zset*(naToOtherRatio-1)+1
        # import ipdb; ipdb.set_trace()
        return Zset / Zset.sum()

    def change_primal_type(self, primal_type):
        assert primal_type in ['est', 'full']
        self.primal_type = primal_type
        self.dv_bench, self.pv_bench = self.violation(np.arange(self.Xres.shape[1]), np.arange(self.Zres.shape[1]))

    def violation(self, remnantX, remnantZ, ord=np.inf):
        covXZ_tmp = self.covXZ[remnantX, :][:, remnantZ]
        covZX_tmp = covXZ_tmp.T
        covXD_tmp = self.covXD[remnantX]
        covZY_tmp = self.covZY[remnantZ]
        dual_violation = np.linalg.norm(covXD_tmp - covXZ_tmp @ scipy.linalg.pinv(covXZ_tmp) @ covXD_tmp, ord=ord)

        if self.primal_type != 'full':
            primal_violation = np.linalg.norm(covZY_tmp - covZX_tmp @ scipy.linalg.pinv(covZX_tmp) @ covZY_tmp, ord=np.inf)
        else:
            ## more accurate primal violation
            covDXDZ_tmp = self.covDXDZ[[0] + [i + 1 for i in remnantX], :][:, [0] + [i + 1 for i in remnantZ]]
            covDZDX_tmp = covDXDZ_tmp.T
            covDZY_tmp = self.covDZY[[0] + [i + 1 for i in remnantZ]]
            primal_violation = np.linalg.norm(covDZY_tmp - covDZDX_tmp @ scipy.linalg.pinv(covDZDX_tmp) @ covDZY_tmp, ord=ord)

        return dual_violation, primal_violation  
    
    def xset_trial(self, it, remnantZ, verbose, gen_next='random'):
        ''' We try to add elements to the X's in random order, while maintaining that the dual
        violation is not violated. Here we use all the Z's, since the dual violation can only
        improve if we add more Z's.
        '''
        np.random.seed(it)
        unusedX = np.arange(self.Xres.shape[1])
        remnantX = []
        while len(unusedX) > 0:
            if gen_next=='random' or remnantX == []:
                next = np.random.choice(len(unusedX), size=1)[0]
            else:
                assert gen_next == 'min_corr'
                p= self.Xres[:,remnantX[-1]] @ self.Xres[:, unusedX] / self.Xres.shape[0]
                p = 1/np.abs(p)
                p /= p.sum()
    #             p = self.Xres[:,remnantX].T @ self.Xres[:, unusedX] / self.Xres.shape[0]
    #             p = 1 / np.abs(p).mean(axis=0)
    #             p = p/p.sum()
                next = np.random.choice(len(unusedX),p=p, size=1)[0]

            dv, pv = self.violation(remnantX + [unusedX[next]], remnantZ)
            if dv < self.dthresh * self.dv_bench:
                remnantX += [unusedX[next]]
            unusedX = np.delete(unusedX, next)

        if remnantX:
            remnantX = np.sort(remnantX)
            if verbose:
                print(remnantX, self.violation(remnantX, remnantZ))
        
            ohe = np.zeros(self.Xres.shape[1]).astype(int)
            ohe[remnantX] = 1
            return ohe
        else:
            return None

    def zset_trial(self, it, remnantX, verbose, gen_next='random'):
        ''' Given a candidate X set, we try to add elements to the Z's in random order,
        while maintaining that the primal violation does not occur.
        '''
        np.random.seed(it)
        
        unusedZ = np.arange(self.Zres.shape[1])
        remnantZ = []
        while len(unusedZ) > 0:
            if gen_next == 'reweight':
                p = self.reweight_Z(unusedZ)
                next = np.random.choice(len(unusedZ),p=p, size=1)[0]
            elif gen_next=='random' or remnantZ == []:
                next = np.random.choice(len(unusedZ), size=1)[0]
            else: 
                assert gen_next == 'min_corr'
                p = self.covZ[remnantZ[-1]][unusedZ]
                p = 1/np.abs(p)
                p /= p.sum()
    #             p = self.Zres[:,remnantZ].T @ self.Zres[:, unusedZ] / self.Zres.shape[0]
    #             p = 1 / np.abs(p).mean(axis=0)
    #             p = p/p.sum()
                next = np.random.choice(len(unusedZ),p=p, size=1)[0]
            dv, pv = self.violation(remnantX, remnantZ + [unusedZ[next]])
            if pv < self.pthresh * self.pv_bench:
                remnantZ += [unusedZ[next]]
            unusedZ = np.delete(unusedZ, next)

        if remnantZ:
            remnantZ = np.sort(remnantZ)
        
            dv, pv = self.violation(remnantX, remnantZ)
            if verbose:
                print(remnantX, remnantZ, dv, pv)
        
            ohe = np.zeros(self.Xres.shape[1] + self.Zres.shape[1]).astype(int)
            ohe[remnantX] = 1
            ohe[self.Xres.shape[1] + remnantZ] = 1
            return ohe
        else:
            return None

    def find_candidate_sets(self, ntrials, verbose=0, n_jobs=-1, 
                            gen_nextZ='random', gen_nextX='random',
                            save_dir='', niters=2, unique_Zsets=[]):
        if unique_Zsets == []:
            unique_Zsets = np.array([np.ones(self.Zres.shape[1])]).astype(int)
        else:
            assert type(unique_Zsets) == np.ndarray and \
                unique_Zsets.dtype == np.int64 and len(unique_Zsets[0] == self.Zres.shape[1])
            
        for j in range(niters):
            # we generate a set of candidate of maximal X sets such that the dual violation does not
            # occur, when we use all the Z's. Note that more Z's can only help the dual.
            candidateX = []
            for remnantZ in tqdm(unique_Zsets):
                remnantZ = np.argwhere(remnantZ).flatten()
                candidateX += Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(self.xset_trial)(it, remnantZ, verbose, gen_nextX)
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
                # self.zset_trial(0, remnantX, verbose, gen_nextZ)
                candidateXZ += Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(self.zset_trial)(it, remnantX, verbose, gen_nextZ)
                                                                for it in range(ntrials))

            candidateXZ = [c for c in candidateXZ if c is not None]

            if not candidateXZ:
                return []

            # this array now contains the one-hot-encodings of the Xset and the Zset (concatenated)
            candidateXZ = np.array(candidateXZ).astype(int)
            # we clean up to keep only unique Zset solutions
            unique_Zsets = np.unique(candidateXZ[:, self.Xres.shape[1]:], axis=0)

            if verbose:
                print(f"Finished iteration {j}!")

        # we clean up to keep only unique pairs of solutions
        unique_XZsets = np.unique(candidateXZ, axis=0)
        # we transform the one hot encodings back to member sets
        final_candidates = []
        for unique_XZ in unique_XZsets:
            Xset = np.argwhere(unique_XZ[:self.Xres.shape[1]]).flatten()
            Zset = np.argwhere(unique_XZ[self.Xres.shape[1]:]).flatten()
            dv, pv = self.violation(Xset, Zset)
            if verbose:
                print(Xset, Zset, dv, pv)
            if pv < self.pthresh * self.pv_bench and dv < self.dthresh * self.dv_bench:
                final_candidates += [(Xset, Zset)]
        
        if verbose:
            print(f"Found {len(final_candidates)} candidates!")
        if save_dir:
            import pickle as pk
            pk.dump(final_candidates, open(save_dir+'candidates.pkl', 'wb'))
        return final_candidates

    def get_estimates(self, candidates, save_dir='', idx_list=[], verbose=0, npass=2):            
        if idx_list == []:
            # Compute and all 
            idx_list = np.arange(len(candidates))
        for idx in tqdm(idx_list):
            fname = ''
            if save_dir:
                fname = save_dir + f'/{idx}/'
                if not os.path.exists(fname):
                    os.mkdir(fname)

            Xset, Zset = candidates[idx]
            est = ProximalDE(semi=True, cv=3, verbose=1, random_state=3)
            est.fit(None, self.Dres, self.Zres[:, Zset], self.Xres[:, Xset], self.Yres)
            sm = est.summary(save_dir=fname)

            svalues, svalues_crit = est.covariance_rank_test(calculate_critical=True)
            rank_pass = np.sum(svalues >= svalues_crit)
            est.run_diagnostics()
            inds = est.influential_set(alpha=0.05)
            inf_set = len(inds)
            
            if verbose:
                df = pd.DataFrame.from_records(sm.tables[2].data)
                header = df.iloc[0] # grab the first row for the header
                df = df[1:] # take the data less the header row
                df.columns = header
                df['pass test'] = df['pass test'].map(lambda x: x == 'True')
                if df['pass test'].sum() > npass:
                    print(f"Inf set size: {inf_set} || # nonzero Cov(XZ) singular values: {rank_pass}")
                    rmXset = np.setdiff1d(np.arange(self.Xres.shape[1]), Xset)
                    print("Kept Xs = ", np.random.choice(self.Xint[Xset],replace=False, size=min(10,len(Xset))))
                    print("Deleted Xs =", np.random.choice(self.Xint[rmXset],replace=False,size=min(10,len(rmXset))))

                    rmZset = np.setdiff1d(np.arange(self.Zres.shape[1]), Zset)
                    print("Kept Zs = ", np.random.choice(self.Zint[Zset],replace=False, size=min(10,len(Zset))))
                    print("Deleted Zs =", np.random.choice(self.Zint[rmZset],replace=False,size=min(10,len(rmZset))))
                    print(est.summary().tables[0], est.summary().tables[2])
            
            
            