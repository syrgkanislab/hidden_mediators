from joblib import Parallel, delayed
import numpy as np
from proximalde.utilities import covariance, svd_critical_value
import scipy.stats
from tqdm import tqdm 
import os 
from proximalde.proximal import ProximalDE
import pandas as pd 

def convert_table_to_df(table):
    df = pd.DataFrame.from_records(table.data)
    header = df.iloc[0] # grab the first row for the header
    if header[0] == '':
        header[0] = 'test'
    df = df[1:] # take the data less the header row
    df.columns = header
    return df
                
class ProxySelection():
    def __init__(self, Xres, Zres, Dres, Yres, violation_type: str = 'est', 
                 primal_type: str = 'full',  est_thresh: float = .1):
        """
        Data-driven algorithm that identifies the maximum set of 
        proxy features Z and X that satisfy dual and primal constraints, 
        and thus produce unbiased valid implicit bias effect estimates 

        Parameters:
            Xres (np.ndarray): Proxies X after controlling for W
            Zres (np.ndarray): Proxies Z after controlling for W
            Dres (np.ndarray): Treatment D after controlling for W
            Yres (np.ndarray): Outcome Y after controlling for W
            violation_type (str): Defines how the primal and dual violations are computed
                at the very end over all candidates.
                Options: 'est' (estimate as proposed in paper; the OLS estimate and thus much quicker) 
                or 'full' (more accurate but more expensive)
            primal_type (str): How the primal is estimated. 
                Options: 'est' (estimate) = is Cov(YZ) in col-span(Cov(Z,X)) or 
                'full' = is Cov(Y,DZ) in col-span(Cov(DZ,DX))
            est_thresh (float): Threshold for estimation-based violations.
        """
        self.Xres = Xres
        self.Zres = Zres 
        self.Dres = Dres
        self.Yres = Yres

        assert violation_type in ['est', 'full']
        self.compute_violation = self.violation_est if violation_type=='est' else self.violation_full
        assert primal_type in ['est', 'full']
        self.primal_type = primal_type
        self.est_thresh = est_thresh
        
        self.covZY = covariance(Zres, Yres)
        self.covXD = covariance(Xres, Dres)
        self.covXZ = covariance(Xres, Zres)

        # replacing covariance with low rank component, cleaning up the noisy eigenvalues
        U, S, Vh = scipy.linalg.svd(self.covXZ, full_matrices=False)
        Scrit = svd_critical_value(Xres, Zres)
        self.XZ_rank = (S > Scrit).sum()
        self.covXZ = U[:, S > Scrit] @ np.diag(S[S > Scrit]) @ Vh[S > Scrit, :]

        # Precompute all relevant covariances and zero-out the statistical zeros
        stderr_covXD = np.sqrt(np.var((Xres - Xres.mean(axis=0)) * (Dres - Dres.mean(axis=0)), axis=0) / Xres.shape[0])
        self.covXD[np.abs(self.covXD).flatten() < 2.6 * stderr_covXD] = 0

        DZres = np.hstack([Dres, Zres])
        self.covDDZ = covariance(DZres, Dres)
        stderr_covDDZ = np.sqrt(np.var((DZres - DZres.mean(axis=0)) * (Dres - Dres.mean(axis=0)), axis=0) / DZres.shape[0])
        self.covDDZ[np.abs(self.covDDZ).flatten() < 2.6 * stderr_covDDZ] = 0

        self.covDXDZ = np.zeros((1 + Xres.shape[1], 1 + Zres.shape[1]))
        self.covDXDZ[1:, 1:] = self.covXZ
        self.covDXDZ[0, :] = self.covDDZ.flatten()
        self.covDXDZ[1:, 0] = self.covXD.flatten()

        stderr_covZY = np.sqrt(np.var((Zres - Zres.mean(axis=0)) * (Yres - Yres.mean(axis=0)), axis=0) / Zres.shape[0])
        self.covZY[np.abs(self.covZY).flatten() < 2.6 * stderr_covZY] = 0

        self.covDZY = np.zeros((1 + Zres.shape[1], 1))
        self.covDZY[1:, :] = self.covZY
        self.covDZY[0, :] = covariance(Dres, Yres).flatten()

        # Baseline primal and dual (estimated) violations to beat = when all of the proxies X and Z are used 
        self.dual_est_bench, self.primal_est_bench = 0, 0
        self.dual_est_bench, self.primal_est_bench = self.violation_est(np.arange(Xres.shape[1]), np.arange(Zres.shape[1]))[:2]

    def violation_est(self, Xidxs, Zidxs):
        """
        Estimates the dual and primal violations 
            (if self.violation_type==est) using OLS solution

        Parameters:
            Xidxs (np.ndarray): Which indices in proxies X to calculate the primal / dual violation for
            Zidxs (np.ndarray): Which indices in proxies Z to calculate the primal / dual violation for

        Returns: 
            Dual estimate violation, primal estimate violation, 
                dual baseline value to compare to, primal baseline value to compare to 
        """
        covXZ_tmp = self.covXZ[Xidxs, :][:, Zidxs]
        covZX_tmp = covXZ_tmp.T
        covXD_tmp = self.covXD[Xidxs]
        covZY_tmp = self.covZY[Zidxs]
        dual_violation = np.linalg.norm(covXD_tmp - covXZ_tmp @ scipy.linalg.pinv(covXZ_tmp) @ covXD_tmp, ord=np.inf)

        if self.primal_type == 'est':
            primal_violation = np.linalg.norm(covZY_tmp - covZX_tmp @ scipy.linalg.pinv(covZX_tmp) @ covZY_tmp, ord=np.inf)
        else: # more accurate primal violation 
            covDXDZ_tmp = self.covDXDZ[[0] + [i + 1 for i in Xidxs], :][:, [0] + [i + 1 for i in Zidxs]]
            covDZDX_tmp = covDXDZ_tmp.T
            covDZY_tmp = self.covDZY[[0] + [i + 1 for i in Zidxs]]
            primal_violation = np.linalg.norm(covDZY_tmp - covDZDX_tmp @ scipy.linalg.pinv(covDZDX_tmp) @ covDZY_tmp, ord=np.inf)

        return dual_violation, primal_violation, self.est_thresh * self.dual_est_bench, self.est_thresh * self.primal_est_bench 
     
    def violation_full(self, Xidxs, Zidxs, return_dual_primal_only = True):
        """
        Accurate computation of dual and primal violation (if self.violation_type==est)

        Parameters:
            Xidxs (np.ndarray): Which indices in proxies X to calculate the primal / dual violation for
            Zidxs (np.ndarray): Which indices in proxies Z to calculate the primal / dual violation for

        Returns: 
            If return_dual_primal_only:
                returns Dual violation, primal violation, 
                dual critical value, primal critical value
            Else:
                returns point estimate and test tables 
        """
        est = ProximalDE(semi=True, cv=3, verbose=1, random_state=3)
        est.fit(None, self.Dres, self.Zres[:, Zidxs], self.Xres[:, Xidxs], self.Yres)
        test_df = convert_table_to_df(est.summary().tables[2])
        if return_dual_primal_only:
            dual_stat, dual_crit = test_df[test_df.test=='dual_violation^3'][['statistic','critical value']].astype(float).iloc[0]
            primal_stat, primal_crit = test_df[test_df.test=='primal_violation^2'][['statistic','critical value']].astype(float).iloc[0]
            return dual_stat, primal_stat, dual_crit, primal_crit
        else:
            point_df = convert_table_to_df(est.summary().tables[0])
            return point_df, test_df
    
    def xset_trial(self, seed, Zidxs):
        """
        Given a candidate Z set (which starts for the first round as all Z indices), 
        we try to add elements to the X's in random order, while maintaining that the dual
        violation is not violated.

        Returns: the one-hot encoding of the candidate Xidxs 
        """
        np.random.seed(seed)
        unusedX = np.arange(self.Xres.shape[1]) #indices we haven't added yet
        Xidxs = [] #indices of X proxies that don't violate the dual with Zidxs
        nfirst_pair_trials = 0 # indicates the first iteration 
        
        while len(unusedX) > 0:
            # For the very first iteration, we will try some nfirst_pair_trials times to find any
            # potential Xidxs (instead of 1, we look for a pair of indices for greater robustness) 
            # that pass the dual estimate
            if nfirst_pair_trials == 0:
                success = False
                while (not success) or (nfirst_pair_trials > 10):
                    next = np.random.choice(len(unusedX), size=min(2, len(unusedX)), replace=False)
                    dual, _, dual_thresh, _ = self.violation_est(Xidxs + unusedX[next].tolist(), Zidxs)
                    if dual < dual_thresh:
                        Xidxs += unusedX[next].tolist()
                        success = True
                    nfirst_pair_trials += 1
                if success:
                    unusedX = np.delete(unusedX, next)
                else:
                    break
            # For all other iterations, we pick a random unused X index to see if adding it to the 
            # already seen Xidxs still passes the dual estimation
            else:
                next = np.random.choice(len(unusedX), size=1, replace=False)
                dual, _, dual_thresh, _ = self.violation_est(Xidxs + unusedX[next].tolist(), Zidxs)
                if dual < dual_thresh:
                    Xidxs += unusedX[next].tolist()
                unusedX = np.delete(unusedX, next)
        print(len(Xidxs))
        # If we found any non-empty set of admissible Xidxs that pass the dual,
        # we will then check the full estimate of the dual and primal to make sure 
        # the candidate Xidxs with the Zidxs don't fail either
        if Xidxs:
            dual, _, dual_thresh, _ = self.compute_violation(Xidxs, Zidxs)
            if dual > dual_thresh:
                return
            
            Xidxs = np.sort(Xidxs)
            ohe = np.zeros(self.Xres.shape[1]).astype(int)
            ohe[Xidxs] = 1
            return ohe
        else:
            return None


    def zset_trial(self, seed, Xidxs):
        """
        Given a candidate X set, we try to add elements to the Z's in random order,
        while maintaining that the primal violation does not occur.
        
        Returns: the one-hot encoding of the candidate Xidxs and Zidxs 
        """
        np.random.seed(seed)
        unusedZ = np.arange(self.Zres.shape[1]) #indices we haven't added yet
        Zidxs = []  #indices of Z proxies that don't violate the primal with Xidxs
        nfirst_pair_trials = 0 # indicates the first iteration
        
        while len(unusedZ) > 0:
            # For the very first iteration, we will try some nfirst_pair_trials times to find any
            # potential Xidxs (instead of 1, we look for a pair of indices for greater robustness) 
            # that pass the dual estimate
            if nfirst_pair_trials == 0:
                success = False
                while (not success) or (nfirst_pair_trials > 10):
                    next = np.random.choice(len(unusedZ), size=min(2, len(unusedZ)), replace=False)
                    _, primal, _, primal_thresh = self.violation_est(Xidxs, Zidxs + unusedZ[next].tolist())
                    if primal < primal_thresh:
                        Zidxs += unusedZ[next].tolist()
                        success = True
                    nfirst_pair_trials += 1
                if success:
                    unusedZ = np.delete(unusedZ, next)
                else:
                    break
            # For all other iterations, we pick a random unused Z index to see if adding it to the 
            # already seen Zidxs still passes the primal estimation.
            else:
                next = np.random.choice(len(unusedZ), size=1, replace=False)
                _, primal, _, primal_thresh = self.violation_est(Xidxs, Zidxs + unusedZ[next].tolist())
                if primal < primal_thresh:
                    Zidxs += unusedZ[next].tolist()
                unusedZ = np.delete(unusedZ, next)
        
        # If we found any non-empty set of admissible Zidxs that pass the primal,
        # we will then check the full estimate of the dual and primal to make sure 
        # the candidate Zidxs with the Xidxs don't fail either
        if Zidxs:
            dual, primal, dual_thresh, primal_thresh = self.compute_violation(Xidxs, Zidxs)
            if dual > dual_thresh or primal > primal_thresh:
                return
            
            Zidxs = np.sort(Zidxs)        
            ohe = np.zeros(self.Xres.shape[1] + self.Zres.shape[1]).astype(int)
            ohe[Xidxs] = 1
            ohe[self.Xres.shape[1] + Zidxs] = 1
            return ohe
        else:
            return None
           
    def find_candidate_sets(self, ntrials, n_jobs=-1, niters=2):
        """
        Finds the candidate proxy sets as indices of X and Z that pass the 
            dual and primal.
        
        ntrials : For each search of X proxies (or Z proxies), we search for ntrials candiates 
            that pass the respective test when paired with the Z candidate (or X respectively)
        n_jobs : number of jobs to parallelize the above trials over 
        niters : Number of rounds to run this (1 round = looks for Xidxs and then Zidxs for 
            each Xidx.
            
        Returns: 
            List of (Xset, Zset) indices that present a candidate that passess the dual / primal
                violations (either the full or the estimate, depending on self.violation_type). 
        """
        
        # For the first round, we intiate with all the Z indices since the dual violation can only
        # improve if we add more Z's.
        unique_Zsets = np.array([np.ones(self.Zres.shape[1])]).astype(int)

        for j in range(niters):
            # we generate a set of candidate of maximal X sets such that the dual violation does not
            # occur, when we use all the Z's. Note that more Z's can only help the dual.
            candidateX = []
            for Zidxs in tqdm(unique_Zsets):
                Zidxs = np.argwhere(Zidxs).flatten()
                candidateX += Parallel(n_jobs=n_jobs)(delayed(self.xset_trial)(it, Zidxs)
                                                                for it in range(ntrials))
            candidateX = [c for c in candidateX if c is not None]
            if not candidateX:
                return []

            candidateX = np.array(candidateX).astype(int)
            # we clean up to keep only the unique solutions
            unique_Xsets = np.unique(candidateX, axis=0)
        
            candidateXZ = []
            
            # for each unique candidate solution of X's
            for Xidxs in tqdm(unique_Xsets):
                Xidxs = np.argwhere(Xidxs).flatten()
                # we try to construct maximal sets of Z's, such that the primal violation
                # does not occur. Note that more X's can only help the primal, which is why
                # we tried to build maximal X's in the first place.
                candidateXZ += Parallel(n_jobs=n_jobs)(delayed(self.zset_trial)(it, Xidxs)
                                                                for it in range(ntrials))

            candidateXZ = [c for c in candidateXZ if c is not None]
            if not candidateXZ:
                return []

            # this array now contains the one-hot-encodings of the Xset and the Zset (concatenated)
            candidateXZ = np.array(candidateXZ).astype(int)
            # we clean up to keep only unique Zset solutions
            unique_Zsets = np.unique(candidateXZ[:, self.Xres.shape[1]:], axis=0)

        # we clean up to keep only unique pairs of solutions
        unique_XZsets = np.unique(candidateXZ, axis=0)
        # we transform the one hot encodings back to member sets
        final_candidates = []
        for unique_XZ in unique_XZsets:
            Xset = np.argwhere(unique_XZ[:self.Xres.shape[1]]).flatten()
            Zset = np.argwhere(unique_XZ[self.Xres.shape[1]:]).flatten()
            final_candidates += [(Xset, Zset)]
        
        return final_candidates
            