import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from proximalde.ukbb_data_utils import load_ukbb_data
from proximalde.ukbb_proximal import ProximalDE_UKBB,residualizeW_ukbb
pd.options.display.max_columns = None
from tqdm import tqdm 
import os 
import argparse 

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run ProximalDE UKBB analysis")
    
    parser.add_argument('--D_label', type=str, default='Female')
    parser.add_argument('--model_regression', type=str, default='linear', 
                        help='Regression model type (default: linear)')
    parser.add_argument('--model_classification', type=str, default='linear', 
                        help='Classification model type (default: linear)')
    parser.add_argument('--Dbin', action='store_true', 
                        help='If set, D is binary')
    parser.add_argument('--Ybin', action='store_true', 
                        help='If set, Y is binary')
    parser.add_argument('--XZbin', action='store_true', 
                        help='If set, X and Z are binary')
    parser.add_argument('--save_addn', type=str, default='', 
                        help='Additional string to append to save filenames')
    
    args = parser.parse_args()

    # Assign parsed arguments to variables
    model_regression = args.model_regression
    model_classification = args.model_classification
    Dbin = args.Dbin
    Ybin = args.Ybin
    XZbin = args.XZbin
    save_addn = args.save_addn
    D_label = args.D_label

    SAVE_PATH = './results/'
    clsf = ''
    if np.any([Dbin, Ybin, XZbin]):
        clsf=f'_Cls={model_classification}'
#     for Y_label in tqdm(['OA', 'myoc','deprs', 'back', 'RA', 'fibro', 'infl', 'copd','chrkd','mgrn','mela','preg', 'endo']):
    for dy in tqdm(['On_dis_RA',  'Obese_OA', 'Black_chrkd']):
            D_label, Y_label = '_'.join(dy.split('_')[:-1]), dy.split('_')[-1]
            W, _, W_feats, X, X_binary, X_feats, Z, Z_binary, Z_feats, Y, D = load_ukbb_data(D_label=D_label, Y_label=Y_label)
            
            if XZbin:
                 X[:,X_binary] += .5
                 Z[:,Z_binary] += .5

            save_dir = f'{SAVE_PATH}/D={D_label}_Y={Y_label}/Dbin={Dbin}_Ybin={Ybin}_XZbin={XZbin}_Rgr={model_regression}{clsf}'
            print(save_dir)
            if not os.path.exists(f'{SAVE_PATH}/D={D_label}_Y={Y_label}/'):
                os.mkdir(f'{SAVE_PATH}/D={D_label}_Y={Y_label}/')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if not os.path.exists(save_dir + '/table1.csv'):
                np.random.seed(4)
                residualizeW_ukbb(W, D, Z, X, Y, D_label=D_label, Y_label=Y_label,
                         model_regression=args.model_regression,
                         save_fname_addn='',
                         binary_D=False, 
                         cv=3, semi=True,
                         n_jobs=-1, verbose=1,
                         random_state=3)
                # est = ProximalDE_UKBB(model_regression=model_regression, model_classification=model_classification, binary_D=Dbin, binary_Y=Ybin,
                #                     binary_X=X_binary if XZbin else [], binary_Z=Z_binary if XZbin else [], 
                #                     semi=True, cv=3, verbose=1, random_state=3)
                
                # est.fit(W, D, Z, X, Y, D_label=D_label, Y_label=Y_label, save_fname_addn=save_addn)                
                # sm = est.summary(decimals=5, save_dir=save_dir)
                # print(sm.tables[0])
                # print(sm.tables[1])
                # print(sm.tables[2])
