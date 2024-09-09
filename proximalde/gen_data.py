import numpy as np
import scipy.special
import pandas as pd 

#############################
### Synthetic data generation 
#############################

def gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g, *, sm=2, sz=1, sx=1, sy=1):
    '''
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
    '''
    W = np.random.normal(0, 1, size=(n, pw))
    D = np.random.binomial(1, scipy.special.expit(2 * W[:, 0]))
    M = a * D + sm * (W[:, 0] + np.random.normal(0, 2, n))
    Z = (e * M + d * D).reshape(-1, 1) + sz * (W[:, [0]] + np.random.normal(0, 1, (n, pz)))
    X = f * M.reshape(-1, 1) + sx * (W[:, [0]] + np.random.normal(0, 1, (n, px)))
    Y = b * M + c * D + g * X[:, 0] + sy * (W[:, 0] + np.random.normal(0, 1, n))
    return W, D, M, Z, X, Y


def gen_data_no_controls(n, pw, pz, px, a, b, c, d, e, f, g, *, sm=2, sz=1, sx=1, sy=1):
    ''' Controls are generated but are irrelevant to the rest
    of the data

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
    '''
    W = np.random.normal(0, 1, size=(n, pw))
    D = np.random.binomial(1, .5 * np.ones(n,))
    M = a * D + sm * np.random.normal(0, 1, (n,))
    Z = (e * M + d * D).reshape(-1, 1) + sz * np.random.normal(0, 1, (n, pz))
    X = f * M.reshape(-1, 1) + sx * np.random.normal(0, 1, (n, px))
    Y = b * M + c * D + g * X[:, 0] + sy * np.random.normal(0, 1, n)
    return W, D, M, Z, X, Y


#############################
### UK Biobank data loading 
#############################


UKBB_DATA_DIR = '/oak/stanford/groups/rbaltman/karaliu/bias_detection/cohort_creation/data/'


def _load_ukbb_data(fname: str, norm: str = 'min_max'):
    """
    
    Loads numpy dataset and corresponding feature list.
    
    Returns 'data' and 'feats'. 
    'data' is a a numpy file of dim = N x #feats from fname, a continuous- and categorical-valued dataset. Note: assumed to be NaN imputed already. 
    'feats' is dim = #feats and the string-valued list of the corresponding feature  names.
    
    Data normalized by 'norm'. Only accepts 'min_max' or '' (pass) right now. 
    If norm = 'min_max':
        normalizes data to [0,1] range, then adjustes by -.5 to be in [-.5, .5].
    

    """
    data = np.load(UKBB_DATA_DIR + f'{fname}_data_rd.npy', allow_pickle=False)    
    feats = np.load(UKBB_DATA_DIR + f'{fname}_feats_rd.npy', allow_pickle=False)

    assert np.isnan(data).sum() == 0, 'NaN values cannot exist in data'
    
    if norm == 'min_max':
        col_min = data.min(axis=0)
        col_max = data.max(axis=0)
        data = (data - col_min) / (col_max - col_min)
        data -= .5
    elif norm == '':
        pass 
    else:
        raise ValueError(f"Data norm type '{norm}' is unknown")
    
    return data, feats
    
def _load_ukbb_W(D_label: str, norm: str):
    """
    
    Loads the confounder / sociodemographic data and feats.
    Note cannot contain the D_label's data, so it makes sure to 
        remove D_label data.
        
    """      
    assert norm == 'min_max'
    W, W_feats = _load_ukbb_data(fname = 'dem')

    all_D_data = np.load(f'{UKBB_DATA_DIR}/potD_data.npy') #already min_max normalized
    all_D_feats = np.load(f'{UKBB_DATA_DIR}/potD_feats.npy')
    
    
    Dlabel_to_fid = {'Female':31, 'Black':21000, 
                     'Obese': 21002, 'Asian': 21000, 
                     'White': 21000, 'Low_inc': 738, 
                     'On_dis': 6146, 'No_uni': 6138, 
                     'No_priv_insr': 4674} #fid is the ID # of a feature in UKBB
    keep_W_idx = [int(f.split('.')[1])!=Dlabel_to_fid[D_label] for f in all_D_feats]
    W = np.concatenate([W, all_D_data[:,keep_W_idx]], axis=1)
    W_feats = np.concatenate([W_feats, all_D_feats[keep_W_idx]])
    return W, W_feats

def load_ukbb_data(D_label: str, Y_label: str, norm: str = 'min_max'):
    """
    
    Loads all UK Biobank data.
    D_label = string of what the D sensitve attribute label is
    Y_label = string of what the Y outcome diagnosis label is
    
    Returns: W, W_feats, X, X_feats, Z, Z_feats, Y, D
        where (W, X, Z, Y, D) are numpy arrays of size N x _
        normalized by 'norm'
        and (W_feats, X_feats, Z_feats) are numpy arrays of the same 
        column size as the corresponding data, describing the data 
        feature names 
        
    """

    Z, Z_feats = _load_ukbb_data(fname = 'srMntSlp', norm = norm)
    X, X_feats = _load_ukbb_data(fname = 'biomMed', norm = norm)
    W, W_feats = _load_ukbb_W(D_label, norm = norm)

    D_df = pd.read_csv(UKBB_DATA_DIR + 'updated_sa_df_pp.csv')
    D = D_df[D_label].to_numpy()     
    Y = pd.read_csv(UKBB_DATA_DIR + 'updated_Y_labels.csv')[Y_label].to_numpy()[:,None] 

    # For female-only diagnosis, only analyze female subppopulation 
    if Y_label in ['endo', 'prg']: 
        assert D_label != 'Female'
        female = D_df['Female'].astype(bool)
        W = W[female]
        Y = Y[female]
        D = D[female]
        Z = Z[female]
        X = X[female]

    return W, W_feats, X, X_feats, Z, Z_feats, Y, D
=======
def gen_data_no_controls_discrete_m(n, pw, pz, px, a, b, c, d, E, F, g, *, sz=1, sx=1, sy=1, pm=1):
    ''' Controls are generated but are irrelevant to the rest
    of the data. Now the mediator is multi-dimensional (takes pm
    non-zero discrete values and zero).

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
    '''
    W = np.random.normal(0, 1, size=(n, pw))
    D = np.random.binomial(1, .5 * np.ones(n,))
    M = np.random.binomial(1, scipy.special.expit(a * (2 * D - 1)))
    M = M.reshape(-1, 1) * np.random.multinomial(1, np.ones(pm) / pm, size=(n,))
    Z = M @ E + d * D.reshape(-1, 1) + sz * np.random.normal(0, 1, (n, pz))
    X = M @ F + sx * np.random.normal(0, 1, (n, px))
    Y = b * np.sum(M, axis=1) + c * D + g * X[:, 0] + sy * np.random.normal(0, 1, n)
    return W, D, M, Z, X, Y
