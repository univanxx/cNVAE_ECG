import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from glob import glob
import wfdb
import ast


def get_ptbxl_database(data_path, ptbxl_path):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Loading and preparing ptbxl_database.csv
    ptbxl_df = pd.read_csv(ptbxl_path + "ptbxl_database.csv")
    ptbxl_df = ptbxl_df[['scp_codes', 'filename_hr', 'age', 'sex', 'height', 'weight', 'device', 'strat_fold', 'static_noise', 'burst_noise']]
    ptbxl_df['filename_hr'] = ptbxl_df['filename_hr'].apply(lambda x: x[x.rfind('/')+1:])
    ptbxl_df["scp_codes"] = ptbxl_df["scp_codes"].apply(lambda x: ast.literal_eval(x))
    ptbxl_df.reset_index(drop=True, inplace=True)
    ##### Processing the scp_codes dictionary field - leave the entry if the probability is greater than 50% #####
    def omit_by(dct, predicate=lambda x: x>50):
        return np.array([k for k in dct.keys() if predicate(dct[k])])
    # Formation of diagnostic fields, each with its own column
    possible_diags = np.unique(np.concatenate(ptbxl_df['scp_codes'].apply(lambda x: omit_by(x)).values))
    res = {key:[] for key in possible_diags}
    for _, row in ptbxl_df.iterrows():
        good_vals = omit_by(row['scp_codes'])
        for key in res.keys():
            if key in good_vals:
                res[key].append(1)
            else:
                res[key].append(0)
    # Adding other diagnoses
    scp_statements = pd.read_csv(ptbxl_path + "scp_statements.csv") 
    for key in scp_statements.iloc[:,0].values:
        if key not in res.keys():
            res[key] = [0] * ptbxl_df.shape[0]

    res = (pd.DataFrame.from_dict(res).rename(columns=dict(zip(scp_statements.iloc[:,0], scp_statements.iloc[:,1]))))
    res = (ptbxl_df.join(res)).iloc[:,1:]
    res.to_csv(data_path + "thirdparty/ptbxl_classes.csv")


def load_wsdb(file):
    signals, fields = wfdb.rdsamp(file.rstrip('.dat'))
    ecg = signals.T.astype('float32')
    leads = np.array(fields["sig_name"])
    sr = fields["fs"]
    ecg[np.isnan(ecg)] = 0.0
    return ecg, leads, sr


##### Uploading PTB-XL to numpy format #####
def ptbxl_to_numpy(data_path, ptbxl_path):
    data_name = 'records500'
    # Loading an ECG and combining it into a dictionary
    dataset = sorted(glob(f'{ptbxl_path}{data_name}/*/*.dat'))
    ptbxl = {}
    # The process of filling the dictionary
    pbar = tqdm(total=len(dataset))
    for i, name in enumerate(dataset):
        ptbxl[name[name.rfind('/')+1:name.rfind('.')]], _, _ = load_wsdb(dataset[i])
        pbar.update(1)
    pbar.close()
    with open(data_path + 'thirdparty/ptbxl.pickle', 'wb') as handle:
        pickle.dump(ptbxl, handle, protocol=pickle.HIGHEST_PROTOCOL)