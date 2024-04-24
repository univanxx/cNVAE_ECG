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
# from imblearn.over_sampling import SMOTE, ADASYN

import sys
sys.path.insert(0, "../data/")
from data_utils import get_ptbxl_database, ptbxl_to_numpy
from patients_filter import patient_filtering


class CVConditional(Dataset):
    def __init__(self, dataset_path, dataset_name,  
                 option='train', type="generate", proportion=None, seed=23):

        self.option = option
        self.proportion = proportion
        assert type in ["generate", "classify"]
        self.type = type
        
        data_path = os.path.join(dataset_path, dataset_name)
        os.makedirs(data_path, exist_ok=True)
        
        self.labels = np.load(os.path.join(data_path,  "labels.npy"))
        self.values = np.load(os.path.join(data_path, "signals.npy"))

        self.res_min = np.load(os.path.join(data_path, "thirdparty", "res_min.npy"))
        self.res_max = np.load(os.path.join(data_path, "thirdparty", "res_max.npy"))

        if self.type != "classify":
            self.values = np.array(self.values)[:, [0,2,6,7,8,9,10,11], :]
        else:
            self.values = np.array(self.values)
            self.res_min = self.get_12_from_8(self.res_min)
            self.res_max = self.get_12_from_8(self.res_max)

        self.values = (self.values - self.res_min) / (self.res_max - self.res_min)
        
        self.ids = np.load(os.path.join(data_path, "thirdparty", f"{option}_ids.npy"))

        self.values = self.values[self.ids,:,:]
        self.labels = self.labels[self.ids,:]

        with open(os.path.join(dataset_path, "label2id.pickle"), 'rb') as f:
            self.label2id = pickle.load(f)

    def __len__(self):
        if self.proportion is not None:
            return int(np.floor(self.ids.shape[0]*self.proportion))
        else:
            return self.ids.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.values[index]).float(), torch.tensor(self.labels[index], dtype=bool)
            
    @staticmethod
    def get_12_from_8(x):
        sec = x[:,0,:] + x[:,1,:]
        avr = -0.5 * (x[:,0,:] + sec)
        avl = 0.5 * (x[:,0,:] - x[:,1,:])
        avf = 0.5 * (sec + x[:,1,:])
        return np.stack([x[:,0,:], sec, x[:,1,:], avr, avl, avf, x[:,2,:], x[:,3,:], x[:,4,:], x[:,5,:], x[:,6,:], x[:,7,:]], axis=1)


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