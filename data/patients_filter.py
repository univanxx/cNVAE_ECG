import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import copy


def init_names(arr1, arr2, arr3, db):
    names = []
    global to_fix
    for filename_i in tqdm(arr1):
        pat_id = db.loc[db.name == filename_i, "patient_id"].values[0]
        if pat_id not in db[db.name.isin(arr2)].patient_id.values and pat_id not in db[db.name.isin(arr3)].patient_id.values:
            names.append(filename_i)
        else:
            to_fix.add(pat_id)
    return names

def patient_filtering(train_arr, val_arr, test_arr, ptbxl_path, labels):

    ptbxl_db = pd.read_csv(ptbxl_path + "ptbxl_database.csv")
    ptbxl_db["name"] = ptbxl_db.filename_hr.apply(lambda x: x[x.rfind('/')+1:])

    global to_fix
    to_fix = set()
    train_names = init_names(train_arr, test_arr, val_arr, ptbxl_db)
    val_names = init_names(val_arr, test_arr, train_arr, ptbxl_db)
    test_names = init_names(test_arr, train_arr, val_arr, ptbxl_db)

    to_fix_names = ptbxl_db[ptbxl_db.patient_id.isin(to_fix)].groupby(["patient_id"])["name"].unique().reset_index()

    new_train_names, new_val_names, new_test_names = copy.deepcopy(train_names), copy.deepcopy(val_names), copy.deepcopy(test_names)

    random.seed(44)
    indexes = random.choices([0,1,2], k=len(to_fix))

    for pat_i, idx in zip(to_fix, indexes):
        names = to_fix_names.loc[to_fix_names.patient_id == pat_i, "name"].values[0]
        if idx == 0:
            new_train_names.extend(names)
        elif idx == 1:
            new_val_names.extend(names)
        else:
            new_test_names.extend(names)

    new_train_names = np.array(new_train_names)
    new_val_names = np.array(new_val_names)
    new_test_names = np.array(new_test_names)

    cur_train_labs, cur_val_labs, cur_test_labs = [], [], []

    for name in tqdm(new_train_names):
        cur_train_labs.append(int(labels.loc[labels.record_name == name, 0].values[0]))

    for name in new_val_names:
        cur_val_labs.append(int(labels.loc[labels.record_name == name, 0].values[0]))

    for name in new_test_names:
        cur_test_labs.append(int(labels.loc[labels.record_name == name, 0].values[0]))

    cur_train_labs = np.array(cur_train_labs)
    cur_val_labs = np.array(cur_val_labs)
    cur_test_labs = np.array(cur_test_labs)

    return new_train_names, cur_train_labs, new_val_names, cur_val_labs, new_test_names, cur_test_labs




