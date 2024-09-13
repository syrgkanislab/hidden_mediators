import json
import numpy as np
import pandas as pd

    
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
    
def _load_ukbb_W(D_label: str, norm: str, W_type: str = ''):
    """
    
    Loads the confounder / sociodemographic data and feats.
    Note cannot contain the D_label's data, so it makes sure to 
        remove D_label data.
        
    """      
    assert norm == 'min_max'
    if W_type != '':
        print(f"Loading {W_type} W!")
        W_type = '_' + W_type
    W, W_feats = _load_ukbb_data(fname = f'dem{W_type}')

    all_D_data = np.load(f'{UKBB_DATA_DIR}/potD_data{W_type}.npy') #already min_max normalized
    all_D_feats = np.load(f'{UKBB_DATA_DIR}/potD_feats{W_type}.npy')
    
    
    Dlabel_to_fid = {'Female':[31], 'Black':[21000], 
                     'Obese': [21002], 'Asian': [21000], 
                     'White': [21000], 'Low_inc': [738, 6138, 6146, 4674], 
                     'On_dis': [6146], 'No_uni': [6138], 
                     'No_priv_insr': [4674]} #fid is the ID # of a feature in UKBB
    keep_W_idx = [int(f.split('.')[1]) not in Dlabel_to_fid[D_label] for f in all_D_feats]
    W = np.concatenate([W, all_D_data[:,keep_W_idx]], axis=1)
    W_feats = np.concatenate([W_feats, all_D_feats[keep_W_idx]])
    return W, W_feats

def load_ukbb_data(D_label: str, Y_label: str, norm: str = 'min_max', W: str= ''):
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
    W, W_feats = _load_ukbb_W(D_label, norm = norm, W_type=W)

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
