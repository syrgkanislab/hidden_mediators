import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from sklearn.model_selection import check_cv
from .crossfit import fit_predict
from .utilities import _check_input
from proximalde.proximal import ProximalDE, estimate_nuisances, second_stage, estimate_final


def _get_ukbb_res_filenames(D_label: str, Y_label: str, save_fname_addn: str = ''):
    """
    """
    if D_label != '' and Y_label != '':
        save_path = f'/oak/stanford/groups/rbaltman/karaliu/bias_detection/causal_analysis/data_hm'

        if Y_label in ['endo', 'preg']:
            save_fname_addn+='_FemOnly'
        D_label = D_label.replace('_', '')
        Winfo = f'_Wrm{D_label}'

        return [f'{save_path}/Yres_{Y_label}{Winfo}{save_fname_addn}', 
            f'{save_path}/Dres_{D_label}{save_fname_addn}',
            f'{save_path}/Xres{Winfo}{save_fname_addn}', 
            f'{save_path}/Zres{Winfo}{save_fname_addn}']
    else:
        return ['']*4


def residualizeW_ukbb(W, D, Z, X, Y, D_label: str, Y_label: str, save_fname_addn: str = '',
                 categorical=True,
                 cv=5, semi=False, multitask=False, n_jobs=-1, verbose=0,
                 random_state=None):
    ''' Residualizes W out of all the other variables using cross-fitting
    and lasso regression models.

    UKBB-specific tool that tries to load each residual first before recomputing it. 
    File names are fixed based on D_label, Y_label, and any additional info under
    save_fname_addn.
    '''
    W, D, Z, X, Y = _check_input(W, D, Z, X, Y)

    if D.shape[1] > 1:
        raise AttributeError("D should be a scalar treatment")
    if Y.shape[1] > 1:
        raise AttributeError("Y should be a scalar outcome")

    if W is None:
        Dres = D - D.mean(axis=0, keepdims=True)
        Zres = Z - Z.mean(axis=0, keepdims=True)
        Xres = X - X.mean(axis=0, keepdims=True)
        Yres = Y - Y.mean(axis=0, keepdims=True)
        splits = None
    else:
        #####
        # Residualizing W out of D, Z, X, Y, using cross-fitting
        # (or semi-cross-fitting) and a Lasso model with regularization
        # chosen via cross-validation
        #####
        cv = check_cv(cv, y=D, classifier=categorical)
        if hasattr(cv, 'shuffle'):
            cv.shuffle = True
        if hasattr(cv, 'random_state'):
            cv.random_state = random_state

        if multitask:
            model = MultiTaskLasso(random_state=random_state)
            modelcv = MultiTaskLassoCV(random_state=random_state)
        else:
            model = Lasso(random_state=random_state)
            modelcv = LassoCV(random_state=random_state)

        splits = list(cv.split(W, D))

        # Need file names to save residuals if data is UKBB
        save_fnames = _get_ukbb_res_filenames(D_label, Y_label, save_fname_addn) 

        #####
        # Try loading each residual and confirming metadata aligns before 
        # recomputing
        #####
        res_list = []
        for path, data in zip(save_fnames, [Y, D, X, Z]):
            current_metadata = np.concatenate([W[splits[0]].mean(axis=0),data[splits[0]].mean(axis=0)])
            try:
                saved_metadata = np.load(f'{path}_meta.npy')
                assert np.all(saved_metadata == current_metadata), "Metadata is not the same"
                res_data = np.load(f'{path}.npy')
                assert (data.shape == res_data.shape), f"Loaded residual shape {res.shape} != current shape {data.shape}"
                print(f"Loaded residual from {path.split('/')[-1]}...") if verbose > 0 else None
            
            except FileNotFoundError:
                print(f"Residualizing {path.split('/')[-1]}...") if verbose > 0 else None
                res_data = data - fit_predict(W, data, modelcv, model, splits, semi, multitask, n_jobs, verbose)
        
                np.save(f'{path}_meta.npy', current_metadata)
                np.save(f'{path}.npy', res_data)
            
            res_list.append(res_data)

    Dres, Zres, Xres, Yres = res_list
    #####
    # Measuring R^2 perfomrance of residualization models (nuisance models)
    #####
    r2D = np.mean(1 - np.mean(Dres**2, axis=0) / np.var(D, axis=0))
    r2Z = np.mean(1 - np.mean(Zres**2, axis=0) / np.var(Z, axis=0))
    r2X = np.mean(1 - np.mean(Xres**2, axis=0) / np.var(X, axis=0))
    r2Y = np.mean(1 - np.mean(Yres**2, axis=0) / np.var(Y, axis=0))

    return Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y, splits


def proximal_direct_effect_ukbb(W, D, Z, X, Y, D_label: str='', Y_label: str='', 
                           save_fname_addn: str = '', 
                           dual_type='Z', ivreg_type='adv',
                           alpha_multipliers=np.array([1.0]), alpha_exponent=0.3,
                           categorical=True, cv=5, semi=True, multitask=False, n_jobs=-1,
                           verbose=0, random_state=None):
    '''
    As in proximal.py but using residualizeW_ukbb.
    '''
    W, D, Z, X, Y = _check_input(W, D, Z, X, Y)

    if D.shape[1] > 1:
        raise AttributeError("D should be a scalar treatment")
    if Y.shape[1] > 1:
        raise AttributeError("Y should be a scalar outcome")

    Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y, splits = \
        residualizeW_ukbb(W, D, Z, X, Y, D_label, Y_label,
                        save_fname_addn=save_fname_addn,
                        categorical=categorical, cv=cv,
                        semi=semi, multitask=multitask,
                        n_jobs=n_jobs, verbose=verbose,
                        random_state=random_state)
        
    point_debiased, std_debiased, idstrength, idstrength_std, point_pre, std_pre, *_ = \
        second_stage(Dres, Zres, Xres, Yres,
                     dual_type=dual_type, ivreg_type=ivreg_type,
                     alpha_multipliers=alpha_multipliers,
                     alpha_exponent=alpha_exponent,
                     cv=cv, n_jobs=n_jobs, verbose=verbose,
                     random_state=random_state)

    # reporting point estimate and standard error of Controlled Direct Effect
    # and R^ performance of nuisance models
    return point_debiased, std_debiased, r2D, r2Z, r2X, r2Y, \
        idstrength, idstrength_std, point_pre, std_pre
    
class ProximalDE_UKBB(ProximalDE):
    ''' Estimate Controlled Direct Effect using Proximal Causal Inference.
        Inherits most functionality from ProximalDE. 
        Main change is the attempt to load residuals before recomputing. 
    '''

    def fit(self, W, D, Z, X, Y,
            D_label: str='', Y_label: str='',
            save_fname_addn: str='',
            Zres_idx = None, Xres_idx=None):
        ''' Train the estimator

        Parameters
        ----------
        W : array (n, pw) or None
            Controls. If None is passed then the rest of the variables are
            simply de-meaned and no residual models are fitted.
        D : array (n, 1) or (n,)
            Treatment
        Z : array (n, pz) or (n,)
            Treatment proxy controls
        X : array (n, px) or (n,)
            Outcome proxy controls
        Y : array (n, 1) or (n,)
            Outcome
        D_label : str
            Treatment label, only used for save / loading data
        Y_label : str
            Outcome label, only used for save / loading data
        save_fname_addn : str
            Optional additional info for saving / loading data
            
        Returns
        -------
        self : object
        '''
        # if diagnostics were previously run after some previous fit then we
        # need to make those diagnostics invalid, since we are refitting
        if hasattr(self, 'diag_'):
            del (self.diag_)

        W, D, Z, X, Y = _check_input(W, D, Z, X, Y)

        if D.shape[1] > 1:
            raise AttributeError("D should be a scalar treatment")
        if Y.shape[1] > 1:
            raise AttributeError("Y should be a scalar outcome")

        # residualize W from all the variables
        Dres, Zres, Xres, Yres, r2D, r2Z, r2X, r2Y, splits = \
            residualizeW_ukbb(W, D, Z, X, Y, D_label=D_label, Y_label=Y_label,
                                save_fname_addn=save_fname_addn,
                                categorical=self.categorical, cv=self.cv,
                                semi=self.semi, multitask=self.multitask,
                                n_jobs=self.n_jobs, verbose=self.verbose,
                                random_state=self.random_state)

        if Zres_idx is not None:
            Zres = Zres[:, Zres_idx]
            print(f"New shape: {Zres.shape}")
        if Xres_idx is not None:
            Xres = Xres[:, Xres_idx]
            print(f"New shape: {Xres.shape}")

        # estimate the nuisance coefficients that solve the moments
        # E[(Yres - eta'Xres - c*Dres) (Dres; Zres)] = 0
        # E[(Dres - gamma'Zres) Xres] = 0
        Dbar, Ybar, eta, gamma, point_pre, std_pre, primal_violation, dual_violation, \
            idstrength, idstrength_std, ivreg_eta, ivreg_gamma, dualIV = \
            estimate_nuisances(Dres, Zres, Xres, Yres,
                               dual_type=self.dual_type, ivreg_type=self.ivreg_type,
                               alpha_multipliers=self.alpha_multipliers,
                               alpha_exponent=self.alpha_exponent,
                               cv=self.cv, n_jobs=self.n_jobs,
                               verbose=self.verbose,
                               random_state=self.random_state)

        # Final moment solution: solve for c the equation
        #   E[(Yres - eta'Xres - c * Dres) (Dres - gamma'Zres)] = 0
        point_debiased, std_debiased, inf = estimate_final(Dbar, Dres, Ybar)

        # Storing fitted parameters and training data as
        # properties of the class
        self.nobs_ = D.shape[0]
        self.pw_ = W.shape[1] if W is not None else 0
        self.pz_ = Zres.shape[1]
        self.px_ = Xres.shape[1]
        self.dual_type_ = self.dual_type
        self.ivreg_type_ = self.ivreg_type
        self.alpha_multipliers_ = self.alpha_multipliers
        self.alpha_exponent_ = self.alpha_exponent
        self.categorical_ = self.categorical
        self.cv_ = self.cv
        self.semi_ = self.semi
        self.multitask_ = self.multitask
        self.W_ = W
        self.D_ = D
        self.Z_ = Z
        self.X_ = X
        self.Y_ = Y
        self.Dres_ = Dres
        self.Zres_ = Zres
        self.Xres_ = Xres
        self.Yres_ = Yres
        self.r2D_ = r2D
        self.r2Z_ = r2Z
        self.r2X_ = r2X
        self.r2Y_ = r2Y
        self.splits_ = splits
        self.Dbar_ = Dbar
        self.Ybar_ = Ybar
        self.eta_ = eta
        self.gamma_ = gamma
        self.point_pre_ = point_pre
        self.stderr_pre_ = std_pre
        self.primal_violation_ = primal_violation
        self.dual_violation_ = dual_violation
        self.point_ = point_debiased
        self.stderr_ = std_debiased
        self.idstrength_ = idstrength
        self.idstrength_std_ = idstrength_std
        self.ivreg_eta_ = ivreg_eta
        self.ivreg_gamma_ = ivreg_gamma
        self.dualIV_ = dualIV
        self.inf_ = inf
        self.D_label_ = D_label
        self.Y_label_ = Y_label

        return self
