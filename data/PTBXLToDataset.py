from torch.utils.data import Dataset
import torch

import pandas as pd
import os
import numpy as np
import pickle
import cv2
from scipy.signal import savgol_filter

from sklearn.model_selection import train_test_split, StratifiedKFold
import glob

from scipy.signal import savgol_filter, filtfilt
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, ADASYN

import sys
sys.path.insert(0, "../data/")
from data_utils import get_ptbxl_database, ptbxl_to_numpy
from patients_filter import patient_filtering


class CVConditional(Dataset):
    def __init__(self, diag_name, size, fold, data_path, 
                 option='train', type="gan_sample", num_folds=5, seed=23, p=0.5, smooth=False, filter=False, ptbxl_path=None):

        self.p = p
        self.option = option
        assert type in ["gan_sample", "gan_no_sample", "classify", "gan_equal"]
        self.type = type
        self.size = size
        self.fold = fold
        self.smooth = smooth
        self.filtered = "filtered" if filter else ''

        os.makedirs(data_path + "thirdparty", exist_ok=True)
        if not os.path.exists(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}"):
            # Get classes
            if not os.path.isfile(data_path+"thirdparty/ptbxl_classes.csv"):
                get_ptbxl_database(data_path, ptbxl_path)
            df_classes = pd.read_csv(data_path+"thirdparty/ptbxl_classes.csv")
            df_classes.rename(columns={"filename_hr": "record_name"}, inplace=True)
            stats = pd.read_csv(ptbxl_path+"scp_statements.csv")
            names = dict(zip(stats.iloc[:,1], stats.iloc[:,5]))
            df = df_classes.rename(columns=names)  
            res = df.loc[:,diag_name].sum(axis=1) > 0
            res = res.to_frame().merge(df_classes["record_name"], left_index=True, right_index=True, how="inner")
            os.makedirs(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}", exist_ok=True)
            res.to_csv(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/{diag_name}_labels.csv", index=False)
            # Get values
            self.data = {}
            ### This file should be a dictionary with paient_id as key value and numpy aray with raw signal as value 
            if not os.path.isfile(data_path+"thirdparty/ptbxl.pickle"):
                ptbxl_to_numpy(data_path, ptbxl_path)
            with open(data_path+"thirdparty/ptbxl.pickle", "rb") as f:
                self.data = pickle.load(f)

            values = np.array(list(self.data.values()))
            values_mod = []
            # filtering
            for file_i in tqdm(values, desc="Filtering signals"):
                transformed_val = CVConditional.transform_frequency(file_i, self.size, 10)
                if self.filtered != '':
                    transformed_val = np.apply_along_axis(lambda x: filtfilt(self.b, self.a, x), axis=1, arr=transformed_val)
                if smooth:
                    values_mod.append(np.apply_along_axis(lambda x: savgol_filter(x, 29, 10), axis=1, arr=transformed_val))
                else:
                    values_mod.append(transformed_val)
                
            values = np.array(values_mod)

            self.res_min, self.res_max = [], []
            for ax in tqdm([0,2,6,7,8,9,10,11]):
                self.res_min.append(np.percentile(values[:,ax,:].min(axis=1), 5))
                self.res_max.append(np.percentile(values[:,ax,:].max(axis=1), 95))
            self.res_min = np.array(self.res_min)[None, :, None]
            self.res_max = np.array(self.res_max)[None, :, None]
            np.save(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/res_min.npy", self.res_min)
            np.save(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/res_max.npy", self.res_max)
 
            values = np.array(list(self.data.values()))
            keys = np.array(list(self.data.keys()))
            min_good = (values[:, [0,2,6,7,8,9,10,11], :] > self.res_min).all(axis=(1,2))
            max_good = (values[:, [0,2,6,7,8,9,10,11], :] < self.res_max).all(axis=(1,2))
            good_keys = keys[min_good & max_good]
            good_keys = np.intersect1d(good_keys, np.array(res.record_name))
            # CV
            skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
            myo = np.array(res.loc[res.record_name.isin(good_keys),0])
            for i, (train_index, test_index) in enumerate(skf.split(good_keys, myo)):
                os.makedirs(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(i) + "_fold", exist_ok=True)

                names_test, names_val, labels_test, labels_val = train_test_split(good_keys[test_index], myo[test_index], 
                                                                                test_size=0.5, random_state=seed, stratify=myo[test_index])
                # Patient filtering
                names_train, labels_train, names_val, labels_val, names_test, labels_test = patient_filtering(good_keys[train_index], names_val, names_test, ptbxl_path, res)
                with open(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(i) + "_fold" + "/train_names.npy", "wb") as f:
                    np.save(f, names_train)
                with open(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(i) + "_fold" + "/train_labels.npy", "wb") as f:
                    np.save(f, labels_train)
                with open(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(i) + "_fold" + "/val_names.npy", "wb") as f:
                    np.save(f, names_val)
                with open(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(i) + "_fold" + "/val_labels.npy", "wb") as f:
                    np.save(f, labels_val)
                with open(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(i) + "_fold" + "/test_names.npy", "wb") as f:
                    np.save(f, names_test)
                with open(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(i) + "_fold" + "/test_labels.npy", "wb") as f:
                    np.save(f, labels_test)

        else:
            print("Found!")

        assert option in ["train", "val", "test"]
        with open(data_path+"thirdparty/ptbxl.pickle", "rb") as f:
            self.data = pickle.load(f)
        self.res_min = np.load(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/res_min.npy")
        self.res_max = np.load(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/res_max.npy")

        if option == 'train':
            with open(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/train_names.npy", "rb") as f:
                self.names = np.load(f, allow_pickle=True)
            self.data_prepared  = []
            for key in tqdm(self.names, desc="Data preparing"):
                transformed_val = CVConditional.transform_frequency(self.data[key], self.size, 10)
                if self.filtered !='':
                    self.data_prepared.append(np.apply_along_axis(lambda x: filtfilt(self.b, self.a, x), axis=1, arr=transformed_val))
                else:
                    self.data_prepared.append(transformed_val)
            with open(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(str(fold)) + "_fold" + "/train_labels.npy", "rb") as f:
                self.labels = np.load(f, allow_pickle=True)
        elif option == 'val':
            with open(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(str(fold)) + "_fold" + "/val_names.npy", "rb") as f:
                self.names = np.load(f, allow_pickle=True)
            with open(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(str(fold)) + "_fold" + "/val_labels.npy", "rb") as f:
                self.labels = np.load(f, allow_pickle=True)
            self.data_prepared  = []
            for key in tqdm(self.names, desc="Data preparing"):
                transformed_val = CVConditional.transform_frequency(self.data[key], self.size, 10)
                if self.filtered !='':
                    self.data_prepared.append(np.apply_along_axis(lambda x: filtfilt(self.b, self.a, x), axis=1, arr=transformed_val))
                else:
                    self.data_prepared.append(transformed_val)
        elif option == 'test':
            with open(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(str(fold)) + "_fold" + "/test_names.npy", "rb") as f:
                self.names = np.load(f, allow_pickle=True)
            with open(data_path+diag_name+f"{self.filtered}_folds_smooth_{self.smooth}/" + str(str(fold)) + "_fold" + "/test_labels.npy", "rb") as f:
                self.labels = np.load(f, allow_pickle=True)
            self.data_prepared  = []
            for key in tqdm(self.names, desc="Data preparing"):
                transformed_val = CVConditional.transform_frequency(self.data[key], self.size, 10)
                if self.filtered !='':
                    self.data_prepared.append(np.apply_along_axis(lambda x: filtfilt(self.b, self.a, x), axis=1, arr=transformed_val))
                else:
                    self.data_prepared.append(transformed_val)

        if self.type != "classify":
            self.data_prepared = np.array(self.data_prepared)[:, [0,2,6,7,8,9,10,11], :]
        else:
            self.data_prepared = np.array(self.data_prepared)

        if self.type == "gan_equal":
            pos_idx = np.where(self.labels == 1)[0]
            neg_idx = np.where(self.labels == 0)[0]
            neg_idx = np.random.choice(neg_idx, self.labels.sum(), replace=False)
            self.data_prepared = self.data_prepared[np.concatenate((neg_idx, pos_idx))]
            np.random.seed(seed)
            np.random.shuffle(self.data_prepared)
            self.labels = np.concatenate((np.zeros_like(neg_idx), np.ones_like(pos_idx)))
            np.random.seed(seed)
            np.random.shuffle(self.labels)

    def __len__(self):
        return self.data_prepared.shape[0]

    def __getitem__(self, index):
            
            if self.type == "gan_sample":
                class_i = np.random.choice(2, 1, p=[self.p, 1-self.p])[0]
                if class_i == 1:
                    idx = np.random.choice(self.labels.nonzero()[0], 1)[0]
                else:
                    idx = np.random.choice((self.labels == 0).nonzero()[0], 1)[0]
                class_i = self.labels[idx]
                sample = self.data_prepared[idx]
            else:
                class_i = self.labels[index]
                sample = self.data_prepared[index]

            if self.smooth:
                sample = np.apply_along_axis(lambda x: savgol_filter(x, 29, 10), axis=1, arr=sample)
                    
            if self.type in ["gan_sample", "gan_no_sample"]:
                sample = (sample - self.res_min.squeeze(axis=0)) / (self.res_max.squeeze(axis=0) - self.res_min.squeeze(axis=0))
                sample[sample > 1] = 1
                sample[sample < 0] = 0
            
            return torch.from_numpy(sample).float(), torch.tensor(class_i).long()
            
            
    @staticmethod
    def transform_frequency(signal, freq, sec):
        resized = cv2.resize(np.array(signal), (freq*sec, signal.shape[0]))
        resized = resized.astype(signal.dtype)
        return resized
    
    @staticmethod
    def get_12_from_8(x):
        sec = x[:,0,:] + x[:,1,:]
        avr = -0.5 * (x[:,0,:] + sec)
        avl = 0.5 * (x[:,0,:] - x[:,1,:])
        avf = 0.5 * (sec + x[:,1,:])
        return np.stack([x[:,0,:], sec, x[:,1,:], avr, avl, avf, x[:,2,:], x[:,3,:], x[:,4,:], x[:,5,:], x[:,6,:], x[:,7,:]], axis=1)
    
    @staticmethod
    def get_norms(norms):
        sec_norm = norms[:,0, :] + norms[:, 1, :]
        avr_norm = -0.5 * (norms[:,0,:] + sec_norm)
        avl_norm = 0.5 * (norms[:,0,:] - norms[:,1,:])
        avf_norm = 0.5 * (sec_norm + norms[:,1,:])
        return np.array([norms[:,0,:], sec_norm, norms[:,1,:], avr_norm, avl_norm, avf_norm, norms[:,2,:], norms[:,3,:], 
                         norms[:,4,:], norms[:,5,:], norms[:,6,:], norms[:,7,:]]).transpose(1,0,2)


class CVGenerated(Dataset):
    def __init__(self, diag_name, model_name, size, fold, data_path, generated_path,
                 proportion=0.1, option='train', seed=23, filter=False, smooth=False):

        self.option = option
        self.type = type
        self.size = size
        self.fold = fold
        self.model_name = model_name
        self.smooth = smooth
        self.filtered = "_filtered" if  filter else ''
        self.filter = filter

        if filter:
            self.res_min = np.load(data_path+f"{diag_name}{self.filtered}_folds_smooth_{self.smooth}/res_min.npy")
            self.res_max = np.load(data_path+f"{diag_name}{self.filtered}_folds_smooth_{self.smooth}/res_max.npy")
        else:
            self.res_min = np.load(data_path+f"{diag_name}_folds_smooth_{self.smooth}/res_min.npy")
            self.res_max = np.load(data_path+f"{diag_name}_folds_smooth_{self.smooth}/res_min.npy")

 
        if not os.path.exists(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold"):
 
            res_ones = self.generate_ecg_together(generated_path+self.model_name, type=1)

            res_ones = res_ones[~np.isnan(res_ones).any(axis=(1,2))]
            labels_ones = np.ones(shape=res_ones.shape[0])
            res_zeros = self.generate_ecg_together(generated_path+self.model_name, type=0)
            res_zeros = res_zeros[~np.isnan(res_zeros).any(axis=(1,2))]
            labels_zeros = np.zeros(shape=res_zeros.shape[0])
            # gan test
            data_test = CVConditional("MI", self.size, fold, data_path, type="classify",
                                        option="test", filter=filter)
                      
            test_ones_ids = np.random.choice(len(labels_ones), data_test.labels.sum(), replace=False)
            test_zeros_ids = np.random.choice(len(labels_zeros), len(data_test) - data_test.labels.sum(), replace=False)

            data = np.vstack([res_ones[test_ones_ids] , res_zeros[test_zeros_ids]])
            labels = np.concatenate((labels_ones[test_ones_ids] , labels_zeros[test_zeros_ids]))
            ids = np.arange(len(labels))
            np.random.seed(seed)
            np.random.shuffle(ids)
            # saving
            os.makedirs(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold")
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/test_ecgs.npy", "wb") as f:
                np.save(f, data[ids])
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/test_labels.npy", "wb") as f:
                np.save(f, labels[ids])  

            other_zeros = np.setdiff1d(np.arange(len(labels_zeros)), test_zeros_ids)
            other_ones = np.setdiff1d(np.arange(len(labels_ones)), test_ones_ids)

            res_zeros = res_zeros[other_zeros]
            res_ones = res_ones[other_ones]
            
            # adding validation
            data_val = CVConditional("MI", self.size, fold, data_path, type="classify",
                                        option="val", filter=filter)

            val_ones_ids = np.random.choice(np.arange(len(res_ones)), data_val.labels.sum(), replace=False)
            val_zeros_ids = np.random.choice(np.arange(len(res_zeros)), len(data_val) - data_val.labels.sum(), replace=False)
            
            data = np.vstack([res_ones[val_ones_ids] , res_zeros[val_zeros_ids]])
            labels = np.concatenate((labels_ones[val_ones_ids] , labels_zeros[val_zeros_ids]))
            ids = np.arange(len(labels))
            np.random.seed(seed)
            np.random.shuffle(ids)
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/val_ecgs.npy", "wb") as f:
                np.save(f, data[ids])
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/val_labels.npy", "wb") as f:
                np.save(f, labels[ids]) 

            other_zeros = np.setdiff1d(np.arange(len(res_zeros)), val_zeros_ids)
            other_ones = np.setdiff1d(np.arange(len(res_ones)), val_ones_ids)

            res_zeros = res_zeros[other_zeros]
            res_ones = res_ones[other_ones]

            # gan train
            data_train = CVConditional("MI", self.size, fold, data_path, type="classify",
                                        option="train", filter=filter)
            
            train_ones_ids = np.random.choice(np.arange(len(res_ones)), data_train.labels.sum(), replace=False)
            train_zeros_ids = np.random.choice(np.arange(len(res_zeros)), len(data_train) - data_train.labels.sum(), replace=False)

            data = np.vstack([res_ones[train_ones_ids] , res_zeros[train_zeros_ids]])
            labels = np.concatenate((labels_ones[train_ones_ids] , labels_zeros[train_zeros_ids]))
            ids = np.arange(len(labels))
            np.random.seed(seed)
            np.random.shuffle(ids)
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/train_ecgs.npy", "wb") as f:
                np.save(f, data[ids])
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/train_labels.npy", "wb") as f:
                np.save(f, labels[ids])  

            other_zeros = np.setdiff1d(np.arange(len(res_zeros)), train_zeros_ids)
            other_ones = np.setdiff1d(np.arange(len(res_ones)), train_ones_ids)

            res_zeros = res_zeros[other_zeros]
            res_ones = res_ones[other_ones]

            # for proportions  
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/ecgs_zeros.npy", "wb") as f:
                np.save(f, res_zeros)
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/ecgs_ones.npy", "wb") as f:
                np.save(f, res_ones)  


        if not os.path.exists(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion)): 

            data_train = CVConditional("MI", self.size, fold, data_path, type="classify",
                                        option="train", filter=filter)
            
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/ecgs_zeros.npy", "rb") as f:
                self.ecgs_zeros = np.load(f, allow_pickle=True)
            labels_zeros = np.zeros(shape=self.ecgs_zeros.shape[0])
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/ecgs_ones.npy", "rb") as f:
                self.ecgs_ones = np.load(f, allow_pickle=True)
            labels_ones = np.ones(shape=self.ecgs_ones.shape[0])    

            train_ones_ids = np.random.choice(len(self.ecgs_ones), int(np.ceil(data_train.labels.sum()*proportion)), replace=False)
            train_zeros_ids = np.random.choice(len(self.ecgs_zeros), int(np.ceil((len(data_train) - data_train.labels.sum())*proportion)), replace=False)
            data = np.vstack([self.ecgs_ones[train_ones_ids] , self.ecgs_zeros[train_zeros_ids]])
            labels = np.concatenate((labels_ones[train_ones_ids] , labels_zeros[train_zeros_ids]))
            ids = np.arange(len(labels))
            np.random.seed(seed)
            np.random.shuffle(ids)
            os.makedirs(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion))
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_ecgs.npy", "wb") as f:
                np.save(f, data[ids])
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_labels.npy", "wb") as f:
                np.save(f, labels[ids])  
        
        assert option in ["train", "test", "val", "proportion"]
        if option == "proportion":
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_ecgs.npy", "rb") as f:
                self.names = np.load(f, allow_pickle=True)
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_labels.npy", "rb") as f:
                self.labels = np.load(f, allow_pickle=True)
        elif option == "test":
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/test_ecgs.npy", "rb") as f:
                self.names = np.load(f, allow_pickle=True)
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/test_labels.npy", "rb") as f:
                self.labels = np.load(f, allow_pickle=True)  
        elif option == "train":
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/train_ecgs.npy", "rb") as f:
                self.names = np.load(f, allow_pickle=True)
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/train_labels.npy", "rb") as f:
                self.labels = np.load(f, allow_pickle=True)  
        else:
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/val_ecgs.npy", "rb") as f:
                self.names = np.load(f, allow_pickle=True)
            with open(data_path+diag_name+'_'+model_name+f"_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/val_labels.npy", "rb") as f:
                self.labels = np.load(f, allow_pickle=True)   

    def __len__(self):
        return self.names.shape[0]

    def __getitem__(self, index):

        sample = PTBXLConditional.transform_frequency(self.names[index], self.size, 10, self.type)
        if self.filter:
            return torch.from_numpy(np.apply_along_axis(lambda x: filtfilt(self.b, self.a, x), axis=1, arr=sample)).float(), torch.tensor(self.labels[index]).long()
        else:
            return torch.from_numpy(sample).float(), torch.tensor(self.labels[index]).long()
            
    @staticmethod 
    def get_12_from_8(x):
        sec = x[:,0,:] + x[:,1,:]
        avr = -0.5 * (x[:,0,:] + sec)
        avl = 0.5 * (x[:,0,:] - x[:,1,:])
        avf = 0.5 * (sec + x[:,1,:])
        return np.stack([x[:,0,:], sec, x[:,1,:], avr, avl, avf, x[:,2,:], x[:,3,:], x[:,4,:], x[:,5,:], x[:,6,:], x[:,7,:]], axis=1)

    def generate_ecg_together(self, path, type, bn=100):
        res = []
        for temp in [0.7, 0.8, 0.9]:
            npzfiles = glob.glob(path + "_fold_" + str(self.fold) + '_temp_' + str(temp) + '_type_' + str(type) + '/bn_' + str(bn) + "/*.npz")
            for npz in tqdm(npzfiles, desc="Loading for temp = {}".format(temp)):
                npzfile = np.load(npz)['samples']
                ecg = npzfile*(self.res_max - self.res_min) + self.res_min
                ecg = CVGenerated.get_12_from_8(ecg)
                res.append(ecg)
        return np.vstack(res)
    

class CVGeneratedOnes(Dataset):
    def __init__(self, diag_name, model_name, size, fold, data_path, generated_path, proportion=0.1, filter=False, smooth=False):

        self.type = type
        self.size = size
        self.fold = fold

        self.model_name = model_name
        self.smooth = smooth
        self.filtered = "_filtered" if  filter else ''
        self.filter = filter

        if filter:
            self.res_min = np.load(data_path+f"{diag_name}{self.filtered}_folds_smooth_{self.smooth}/res_min.npy")
            self.res_max = np.load(data_path+f"{diag_name}{self.filtered}_folds_smooth_{self.smooth}/res_max.npy")
        else:
            self.res_min = np.load(data_path+f"{diag_name}_folds_smooth_{self.smooth}/res_min.npy")
            self.res_max = np.load(data_path+f"{diag_name}_folds_smooth_{self.smooth}/res_min.npy")


        if not os.path.exists(data_path+diag_name+'_'+model_name+f"_generated_ones{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion)):  
            data_train = CVConditional("MI", 2560 // 10, fold, data_path, type="classify",
                                        option="train")
            
            self.ecgs_ones = self.generate_ecg_together(generated_path+self.model_name, fold=fold, type=0)
            self.ecgs_ones = self.ecgs_ones[~np.isnan(self.ecgs_ones).any(axis=(1,2))] 

            how_many = int(np.ceil((len(data_train) - data_train.labels.sum() - data_train.labels.sum())*proportion))
            train_ones_ids = np.random.choice(len(self.ecgs_ones), how_many, replace=False)

            data = self.ecgs_ones[train_ones_ids]
            labels = np.ones(shape=how_many)  

            os.makedirs(data_path+diag_name+'_'+model_name+f"_generated_ones{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion))
            with open(data_path+diag_name+'_'+model_name+f"_generated_ones{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_ecgs.npy", "wb") as f:
                np.save(f, data)
            with open(data_path+diag_name+'_'+model_name+f"_generated_ones{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_labels.npy", "wb") as f:
                np.save(f, labels)  
        
        with open(data_path+diag_name+'_'+model_name+f"_generated_ones{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_ecgs.npy", "rb") as f:
            self.names = np.load(f, allow_pickle=True)
        with open(data_path+diag_name+'_'+model_name+f"_generated_ones{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_labels.npy", "rb") as f:
            self.labels = np.load(f, allow_pickle=True)    

    def __len__(self):
        return self.names.shape[0]

    def __getitem__(self, index):
        sample = PTBXLConditional.transform_frequency(self.names[index], self.size, 10, self.type)
        return torch.from_numpy(sample).float(), torch.tensor(self.labels[index]).long()

    def generate_ecg_together(self, path, fold=0, type=0, bn=100):
        res = []
        for temp in [0.7, 0.8, 0.9]:
            npzfiles = glob.glob(path + "_fold_" + str(fold) + '_temp_' + str(temp) + '_type_' + str(type) + '/bn_' + str(bn) + "/*.npz")
            for npz in npzfiles:
                npzfile = np.load(npz)['samples']
                ecg = npzfile*(self.res_max - self.res_min) + self.res_min
                ecg = CVGenerated.get_12_from_8(ecg)
                res.append(ecg)
        return np.vstack(res)
    

class CVGeneratedGans(Dataset):
    def __init__(self, diag_name, size, fold, data_path, generated_path, model_type,
                 proportion=0.1, option='train', seed=23, filter=False, smooth=False):

        self.option = option
        self.type = type
        self.size = size
        self.fold = fold
        self.smooth = smooth
        self.filtered = "_filtered" if  filter else ''
        self.filter = filter
        self.model_type = model_type
        assert self.model_type in ["p2p", "w_star"]
        self.generated_path = generated_path

        if filter:
            self.res_min = np.load(data_path+f"{diag_name}{self.filtered}_folds_smooth_{self.smooth}/res_min.npy")
            self.res_max = np.load(data_path+f"{diag_name}{self.filtered}_folds_smooth_{self.smooth}/res_max.npy")
        else:
            self.res_min = np.load(data_path+f"{diag_name}_folds_smooth_{self.smooth}/res_min.npy")
            self.res_max = np.load(data_path+f"{diag_name}_folds_smooth_{self.smooth}/res_min.npy")

 
        if not os.path.exists(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold"):
 
            res_ones = self.generate_ecg_together(type=1)
            res_ones = res_ones[~np.isnan(res_ones).any(axis=(1,2))]
            labels_ones = np.ones(shape=res_ones.shape[0])

            res_zeros = self.generate_ecg_together(type=0)
            res_zeros = res_zeros[~np.isnan(res_zeros).any(axis=(1,2))]
            labels_zeros = np.zeros(shape=res_zeros.shape[0])
            # gan test
            data_test = CVConditional("MI", self.size, fold, data_path, type="classify",
                                        option="test", filter=filter)
                      
            test_ones_ids = np.random.choice(len(labels_ones), data_test.labels.sum(), replace=False)
            test_zeros_ids = np.random.choice(len(labels_zeros), len(data_test) - data_test.labels.sum(), replace=False)

            data = np.vstack([res_ones[test_ones_ids] , res_zeros[test_zeros_ids]])
            labels = np.concatenate((labels_ones[test_ones_ids] , labels_zeros[test_zeros_ids]))
            ids = np.arange(len(labels))
            np.random.seed(seed)
            np.random.shuffle(ids)
            # saving
            os.makedirs(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold")
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/test_ecgs.npy", "wb") as f:
                np.save(f, data[ids])
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/test_labels.npy", "wb") as f:
                np.save(f, labels[ids])  

            other_zeros = np.setdiff1d(np.arange(len(labels_zeros)), test_zeros_ids)
            other_ones = np.setdiff1d(np.arange(len(labels_ones)), test_ones_ids)

            res_zeros = res_zeros[other_zeros]
            res_ones = res_ones[other_ones]
            
            # adding validation
            data_val = CVConditional("MI", self.size, fold, data_path, type="classify",
                                        option="val", filter=filter)

            val_ones_ids = np.random.choice(np.arange(len(res_ones)), data_val.labels.sum(), replace=False)
            val_zeros_ids = np.random.choice(np.arange(len(res_zeros)), len(data_val) - data_val.labels.sum(), replace=False)
            
            data = np.vstack([res_ones[val_ones_ids] , res_zeros[val_zeros_ids]])
            labels = np.concatenate((labels_ones[val_ones_ids] , labels_zeros[val_zeros_ids]))
            ids = np.arange(len(labels))
            np.random.seed(seed)
            np.random.shuffle(ids)
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/val_ecgs.npy", "wb") as f:
                np.save(f, data[ids])
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/val_labels.npy", "wb") as f:
                np.save(f, labels[ids]) 

            other_zeros = np.setdiff1d(np.arange(len(res_zeros)), val_zeros_ids)
            other_ones = np.setdiff1d(np.arange(len(res_ones)), val_ones_ids)

            res_zeros = res_zeros[other_zeros]
            res_ones = res_ones[other_ones]

            # gan train
            data_train = CVConditional("MI", self.size, fold, data_path, type="classify",
                                        option="train", filter=filter)
            
            train_ones_ids = np.random.choice(np.arange(len(res_ones)), data_train.labels.sum(), replace=False)
            train_zeros_ids = np.random.choice(np.arange(len(res_zeros)), len(data_train) - data_train.labels.sum(), replace=False)

            data = np.vstack([res_ones[train_ones_ids] , res_zeros[train_zeros_ids]])
            labels = np.concatenate((labels_ones[train_ones_ids] , labels_zeros[train_zeros_ids]))
            ids = np.arange(len(labels))
            np.random.seed(seed)
            np.random.shuffle(ids)
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/train_ecgs.npy", "wb") as f:
                np.save(f, data[ids])
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/train_labels.npy", "wb") as f:
                np.save(f, labels[ids])  

            other_zeros = np.setdiff1d(np.arange(len(res_zeros)), train_zeros_ids)
            other_ones = np.setdiff1d(np.arange(len(res_ones)), train_ones_ids)

            res_zeros = res_zeros[other_zeros]
            res_ones = res_ones[other_ones]

            # for proportions  
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/ecgs_zeros.npy", "wb") as f:
                np.save(f, res_zeros)
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/ecgs_ones.npy", "wb") as f:
                np.save(f, res_ones)  


        if not os.path.exists(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion)): 

            data_train = CVConditional("MI", self.size, fold, data_path, type="classify",
                                        option="train", filter=filter)
            
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/ecgs_zeros.npy", "rb") as f:
                self.ecgs_zeros = np.load(f, allow_pickle=True)
            labels_zeros = np.zeros(shape=self.ecgs_zeros.shape[0])
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/ecgs_ones.npy", "rb") as f:
                self.ecgs_ones = np.load(f, allow_pickle=True)
            labels_ones = np.ones(shape=self.ecgs_ones.shape[0])    

            train_ones_ids = np.random.choice(len(self.ecgs_ones), int(np.ceil(data_train.labels.sum()*proportion)), replace=False)
            train_zeros_ids = np.random.choice(len(self.ecgs_zeros), int(np.ceil((len(data_train) - data_train.labels.sum())*proportion)), replace=False)
            data = np.vstack([self.ecgs_ones[train_ones_ids] , self.ecgs_zeros[train_zeros_ids]])
            labels = np.concatenate((labels_ones[train_ones_ids] , labels_zeros[train_zeros_ids]))
            ids = np.arange(len(labels))
            np.random.seed(seed)
            np.random.shuffle(ids)
            os.makedirs(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion))
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_ecgs.npy", "wb") as f:
                np.save(f, data[ids])
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_labels.npy", "wb") as f:
                np.save(f, labels[ids])  
        
        assert option in ["train", "test", "val", "proportion"]
        if option == "proportion":
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_ecgs.npy", "rb") as f:
                self.names = np.load(f, allow_pickle=True)
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_labels.npy", "rb") as f:
                self.labels = np.load(f, allow_pickle=True)
        elif option == "test":
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/test_ecgs.npy", "rb") as f:
                self.names = np.load(f, allow_pickle=True)
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/test_labels.npy", "rb") as f:
                self.labels = np.load(f, allow_pickle=True)  
        elif option == "train":
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/train_ecgs.npy", "rb") as f:
                self.names = np.load(f, allow_pickle=True)
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/train_labels.npy", "rb") as f:
                self.labels = np.load(f, allow_pickle=True)  
        else:
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/val_ecgs.npy", "rb") as f:
                self.names = np.load(f, allow_pickle=True)
            with open(data_path+diag_name+'_'+model_type+f"_gans_generated{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/val_labels.npy", "rb") as f:
                self.labels = np.load(f, allow_pickle=True)   

    def __len__(self):
        return self.names.shape[0]

    def __getitem__(self, index):

        sample = PTBXLConditional.transform_frequency(self.names[index], self.size, 10, self.type)
        if self.filter:
            return torch.from_numpy(np.apply_along_axis(lambda x: filtfilt(self.b, self.a, x), axis=1, arr=sample)).float(), torch.tensor(self.labels[index]).long()
        else:
            return torch.from_numpy(sample).float(), torch.tensor(self.labels[index]).long()
            
    @staticmethod 
    def get_12_from_8(x):
        sec = x[:,0,:] + x[:,1,:]
        avr = -0.5 * (x[:,0,:] + sec)
        avl = 0.5 * (x[:,0,:] - x[:,1,:])
        avf = 0.5 * (sec + x[:,1,:])
        return np.stack([x[:,0,:], sec, x[:,1,:], avr, avl, avf, x[:,2,:], x[:,3,:], x[:,4,:], x[:,5,:], x[:,6,:], x[:,7,:]], axis=1)

    def generate_ecg_together(self, type):
        res = []
        npzfiles = glob.glob(self.generated_path + "results_{}/fold_{}_type_{}/*".format(self.model_type, self.fold, type))
        for npz in tqdm(npzfiles):
            npzfile = np.load(npz)['samples']
            ecg = npzfile*(self.res_max - self.res_min) + self.res_min
            ecg = CVGenerated.get_12_from_8(ecg)
            res.append(ecg)
        return np.vstack(res)


class CVGeneratedOnesGans(Dataset):
    def __init__(self, diag_name, model_type, size, fold, data_path, generated_path, 
                 proportion=0.1, filter=False, smooth=False):

        self.type = type
        self.size = size
        self.fold = fold
        self.smooth = smooth
        self.filtered = "_filtered" if  filter else ''
        self.filter = filter
        self.model_type = model_type
        assert self.model_type in ["p2p", "w_star"]
        self.generated_path = generated_path

        if filter:
            self.res_min = np.load(data_path+f"{diag_name}{self.filtered}_folds_smooth_{self.smooth}/res_min.npy")
            self.res_max = np.load(data_path+f"{diag_name}{self.filtered}_folds_smooth_{self.smooth}/res_max.npy")
        else:
            self.res_min = np.load(data_path+f"{diag_name}_folds_smooth_{self.smooth}/res_min.npy")
            self.res_max = np.load(data_path+f"{diag_name}_folds_smooth_{self.smooth}/res_min.npy")

        if not os.path.exists(data_path+diag_name+'_'+model_type+f"_generated_ones_gans{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion)):  
            data_train = CVConditional("MI", 2560 // 10, fold, data_path, type="classify",
                                        option="train")
            
            self.ecgs_ones = self.generate_ecg_together(type=0)
            self.ecgs_ones = self.ecgs_ones[~np.isnan(self.ecgs_ones).any(axis=(1,2))] 

            how_many = int(np.ceil((len(data_train) - data_train.labels.sum() - data_train.labels.sum())*proportion))
            train_ones_ids = np.random.choice(len(self.ecgs_ones), how_many, replace=False)

            data = self.ecgs_ones[train_ones_ids]
            labels = np.ones(shape=how_many)  

            os.makedirs(data_path+diag_name+'_'+model_type+f"_generated_ones_gans{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion))
            with open(data_path+diag_name+'_'+model_type+f"_generated_ones_gans{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_ecgs.npy", "wb") as f:
                np.save(f, data)
            with open(data_path+diag_name+'_'+model_type+f"_generated_ones_gans{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_labels.npy", "wb") as f:
                np.save(f, labels)  
        
        with open(data_path+diag_name+'_'+model_type+f"_generated_ones_gans{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_ecgs.npy", "rb") as f:
            self.names = np.load(f, allow_pickle=True)
        with open(data_path+diag_name+'_'+model_type+f"_generated_ones_gans{self.filtered}_folds_smooth_{self.smooth}/" + str(fold) + "_fold" + "/" + str(proportion) + "/train_labels.npy", "rb") as f:
            self.labels = np.load(f, allow_pickle=True)    

    def __len__(self):
        return self.names.shape[0]

    def __getitem__(self, index):
        sample = PTBXLConditional.transform_frequency(self.names[index], self.size, 10, self.type)
        return torch.from_numpy(sample).float(), torch.tensor(self.labels[index]).long()

    def generate_ecg_together(self, type):
        res = []
        npzfiles = glob.glob(self.generated_path + "results_{}/fold_{}_type_{}/*".format(self.model_type, self.fold, type))
        for npz in tqdm(npzfiles):
            npzfile = np.load(npz)['samples']
            ecg = npzfile*(self.res_max - self.res_min) + self.res_min
            ecg = CVGenerated.get_12_from_8(ecg)
            res.append(ecg)
        return np.vstack(res)