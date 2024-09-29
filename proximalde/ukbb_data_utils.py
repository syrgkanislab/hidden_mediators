import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

    
def get_coding_dict(fids):
    coding_dir = '/oak/stanford/groups/rbaltman/hmccann/codings_2022/'
    coding_dict_ = json.load(open(f'{coding_dir}single_categoricalcodings.json', 'r'))
    coding_dict_.update(json.load(open(f'{coding_dir}multiple_categoricalcodings.json', 'r')))

    coding_dict = {} # maps fid: {cat label: string label, ...}
    for f in fids:
        assert str(f) in coding_dict_
        
        coding = coding_dict_[f][0]
        df = pd.read_csv(f'{coding_dir}{coding}.csv', sep='\t')
        dct = dict(zip(df.Coding, df.Meaning))
        coding_dict[f] = dct
    return coding_dict

def get_int_feats(feats):
    names = [] 
    for var in ['sc', 'mc', 'intg', 'cont']: 
        names.append(pd.read_csv(f'/oak/stanford/groups/rbaltman/karaliu/bias_detection/cohort_creation/helper_data/{var}.csv', sep='\t'))
        names[-1]['var'] = var
    names = pd.concat(names)
    coding_dict = get_coding_dict(np.array([x.split('.')[1] for x in feats if x.count('.')>1]))
    all_fids = np.array([x.split('.')[1] for x in feats])
    label_meanings = []
    for x in feats:
        if x.count('.') > 1:
            label_meanings.append('='+coding_dict[str(x.split('.')[1])][int(x.split('.')[2])])
        else:
            label_meanings.append('')
    dscrs = np.array([names[names['Field ID'].astype(str) == f].Description.iloc[0] for f in all_fids])
    return np.array([f'{x}{y}' for x,y in zip(dscrs, label_meanings)])

def is_matrix_binary(data):
    """
    Data is a N x M numpy array that is NOT standardized / normalized,
    containing a mix of M binary and numerical attribures. 
    Returns a M dimensional boolean array where True flags the feature
    as binary."""
    is_binary = np.all((data == 0) | (data == 1), axis=0)
    return is_binary

#############################
### UK Biobank data loading 
#############################


UKBB_DATA_DIR = '/oak/stanford/groups/rbaltman/karaliu/bias_detection/cohort_creation/data/'

def load_ukbb_XZ_data():

    def _load_data(fname: str):
        data = np.load(UKBB_DATA_DIR + f'{fname}_data_rd.npy', allow_pickle=False)    
        feats = np.load(UKBB_DATA_DIR + f'{fname}_feats_rd.npy', allow_pickle=False)
        assert np.isnan(data).sum() == 0, 'NaN values cannot exist in data'
        return data, feats
    
    Z, Z_feats = _load_data(fname = 'srMntSlp')
    X, X_feats = _load_data(fname = 'biomMed')
    return X, X_feats, Z, Z_feats


def load_ukbb_res_data(D_label, Y_label):
    print("Assuming D,Y,Z all treated as continuous, using linear regression of W")
    _get_path = lambda fname: f'/oak/stanford/groups/rbaltman/karaliu/bias_detection/causal_analysis/data_hm_std/{fname}'
    D_label = D_label.replace('_', '')
    save_fname_addn = ''
    if Y_label in ['endo', 'preg']:
        save_fname_addn='_FemOnly'
    Winfo = f'_Wrm{D_label}'
    Yres = np.load(_get_path(f'Yres_{Y_label}{Winfo}{save_fname_addn}_Rgrs=linear.npy')) 
    Dres = np.load(_get_path(f'Dres_{D_label}{save_fname_addn}_Rgrs=linear.npy')) 
    Xres = np.load(_get_path(f'Xres{Winfo}{save_fname_addn}_Rgrs=linear.npy')) 
    Zres = np.load(_get_path(f'Zres{Winfo}{save_fname_addn}_Rgrs=linear.npy')) 
    return Xres, Zres, Yres, Dres



def _preprocess_data(data, is_binary):
    data[:,is_binary] = data[:,is_binary] - .5 
    data[:,~is_binary] = StandardScaler().fit_transform(data[:,~is_binary])
    return data 

def _load_ukbb_data(fname: str):
    """
    
    Loads numpy dataset and corresponding feature list.
    
    Returns: data, is_binary, feats. 
    'data': numpy file of dim = N x #feats from fname, a continuous- and categorical-valued dataset. 
        Note: assumed to be NaN imputed already. 
    'is_binary': bool numpy array of dim = #feats, True if that feature is binary
    'feats': dim = #feats and the string-valued list of the corresponding feature  names.
    
    Centers binary data and standard scales continuous data. 

    """
    data = np.load(UKBB_DATA_DIR + f'{fname}_data_rd.npy', allow_pickle=False)    
    feats = np.load(UKBB_DATA_DIR + f'{fname}_feats_rd.npy', allow_pickle=False)
    assert np.isnan(data).sum() == 0, 'NaN values cannot exist in data'
    
    is_binary = is_matrix_binary(data)
    data = _preprocess_data(data, is_binary)
    return data, is_binary, feats
    
def _load_ukbb_W(D_label: str):
    """
    
    Loads the confounder / sociodemographic data and feats.
    Note cannot contain the D_label's data, so it makes sure to 
        remove D_label data.
        
    """      
    W, W_binary, W_feats = _load_ukbb_data(fname = f'dem')

    all_D_data = np.load(f'{UKBB_DATA_DIR}/potD_data.npy')
    all_D_feats = np.load(f'{UKBB_DATA_DIR}/potD_feats.npy')
    all_D_binary = is_matrix_binary(all_D_data)
    all_D_data = _preprocess_data(all_D_data, all_D_binary)
    
    Dlabel_to_fid = {'Female':[31], 'Black':[21000], 
                     'Obese': [21002], 'Asian': [21000], 
                     'White': [21000], 'Low_inc': [738, 6138, 6146], 
                     'On_dis': [6146, 6138, 738], 'No_uni': [6138, 738, 6146], 
                     'No_priv_insr': [4674]} #fid is the ID # of a feature in UKBB
    keep_W_idx = [int(f.split('.')[1]) not in Dlabel_to_fid[D_label] for f in all_D_feats]
    W = np.concatenate([W, all_D_data[:,keep_W_idx]], axis=1)
    W_binary = np.concatenate([W_binary, all_D_binary[keep_W_idx]])
    W_feats = np.concatenate([W_feats, all_D_feats[keep_W_idx]])
    return W, W_binary, W_feats
    
def load_ukbb_data(D_label: str, Y_label: str):
    """
    
    Loads all UK Biobank data.
    D_label = string of what the D sensitve attribute label is
    Y_label = string of what the Y outcome diagnosis label is
    
    Returns: W, W_feats, X, X_feats, Z, Z_feats, Y, D
        where (W, X, Z, Y, D) are numpy arrays of size N x _
        and (W_feats, X_feats, Z_feats) are numpy arrays of the same 
        column size as the corresponding data, describing the data 
        feature names. 
    Centers binary data and standard scales continuous data. 

        
    """

    Z, Z_binary, Z_feats = _load_ukbb_data(fname = 'srMntSlp')
    X, X_binary, X_feats = _load_ukbb_data(fname = 'biomMed')
    
    W, W_binary, W_feats = _load_ukbb_W(D_label)

    D_df = pd.read_csv(UKBB_DATA_DIR + 'updated_sa_df_pp.csv')
    D = D_df[D_label].to_numpy()     
    Y = pd.read_csv(UKBB_DATA_DIR + 'updated_Y_labels.csv')[Y_label].to_numpy()[:,None] 
    # For female-only diagnosis, only analyze female subppopulation 
    if Y_label in ['endo', 'preg']: 
        assert D_label != 'Female'
        female = D_df['Female'].astype(bool)
        W = W[female]
        Y = Y[female]
        D = D[female]
        Z = Z[female]
        X = X[female]

    return W, W_binary, W_feats, X, X_binary, X_feats, Z, Z_binary, Z_feats, Y, D

def load_synthetic_data(norm='minmax'):
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
    from proximalde.gen_data import gen_data_complex
    np.random.seed(0)
    a = 1.0  # a*b is the indirect effect through mediator
    b = 1.0
    c = .5  # this is the direct effect we want to estimate
    d = .1  # this can be zero; does not hurt
    e = .4  # if the product of e*f is small, then we have a weak instrument
    f = .4  # if the product of e*f is small, then we have a weak instrument
    g = .1  # this can be zero; does not hurt
    n = 50000
    pw = 10
    pz, px = 20, 20
    W, D, _, Z, X, Y = gen_data_complex(n, pw, pz, px, a, b, c, d, e, f, g)
    def minmax(data):
        col_min = data.min(axis=0)
        col_max = data.max(axis=0)
        data = (data - col_min) / (col_max - col_min)
        data -= .5
        return data
    if norm == 'minmax':
        W, Z, X = [minmax(x) for x in [W, Z, X ]]
    return W, None, X, None, Z, None, Y[:,None], D[:,None]