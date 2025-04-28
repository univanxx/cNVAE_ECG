from torch.utils.data import Dataset
import torch
import os
import numpy as np
import sys
sys.path.insert(0, "../data/")


class ECGDataset(Dataset):
    def __init__(self, dataset_path, dataset_name, label2id, selected_classes,
                 option='train', type="generate"):

        self.option = option

        assert type in ["classify", "generate"]
        self.type = type
        
        data_path = os.path.join(dataset_path, dataset_name)
        os.makedirs(data_path, exist_ok=True)
        
        self.labels = np.load(os.path.join(data_path,  "labels.npy"))
        self.values = np.load(os.path.join(data_path, "signals.npy"))

        self.res_min = np.load(os.path.join(data_path, "thirdparty", "res_min.npy"))
        self.res_max = np.load(os.path.join(data_path, "thirdparty", "res_max.npy"))

        if self.type != "classify":
            # Select independent leads
            self.values = np.array(self.values)[:, [0,2,6,7,8,9,10,11], :]
            # Min-max normalization
            self.values = (self.values - self.res_min) / (self.res_max - self.res_min)
        else:
            self.values = np.array(self.values)

        selected = [label2id[class_i] for class_i in selected_classes]
        
        self.ids = np.load(os.path.join(data_path, "thirdparty", f"{option}_ids.npy"))
        self.values = self.values[self.ids,:,:]

        self.labels = self.labels[self.ids, :]
        self.labels = self.labels[:, selected]

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.values[index]).float(), torch.tensor(self.labels[index], dtype=bool)


class GeneratedECGDataset(Dataset):
    def __init__(self, labels_path, generated_path, model_name, task_name, label2id, selected_classes, proportion):

        self.model_name = model_name
        assert model_name in ["p2p", "wg*", "nvae"]
        
        labels = np.load(labels_path)[np.load(os.path.dirname(labels_path)+"/thirdparty/train_ids.npy")]
        selected = [label2id[class_i] for class_i in selected_classes]

        generated_signals = []
        generated_labels = []

        if task_name == "addition":
            # Adding generated signals with proportion
            unique_labels, labels_numbers = np.unique(labels, axis=0, return_counts=True)
            for i, (label_id, label_num) in enumerate(zip(unique_labels, labels_numbers)):
                required_num = int(np.round(label_num * proportion))
                if required_num != 0:
                    samples_added, file_num, new_signals  = 0, 0, []
                    while samples_added < required_num:
                        signals = np.load(os.path.join(generated_path, model_name, task_name) + f"/label_{i}_sample_{file_num}.npy")
                        samples_added += signals.shape[0]
                        new_signals.append(signals)
                    new_signals = np.vstack(new_signals)[:required_num]
                    generated_signals.append(new_signals)
                    generated_labels.append(np.vstack([label_id] * required_num))
                    assert new_signals.shape[0] == required_num
                    assert generated_labels[-1].shape[0] == required_num
        elif task_name == "imbalance":
            # Adding generated signals to mitigate class imbalance
            count = np.max(labels.sum(axis=0)) - labels.sum(axis=0)[1:]
            for i, count_i in enumerate(count):
                required_num = int(np.round(count_i * proportion))
                samples_added, file_num, new_signals  = 0, 0, []
                while samples_added < required_num:
                    signals = np.load(os.path.join(generated_path, model_name, task_name) + f"/label_{i}_sample_{file_num}.npy")
                    samples_added += signals.shape[0]
                    new_signals.append(signals)
                new_signals = np.vstack(new_signals)[:required_num]
                generated_signals.append(new_signals)
                label_id = np.zeros(9)
                label_id[1+i] = 1
                generated_labels.append(np.vstack([label_id] * required_num))
                assert new_signals.shape[0] == required_num
                assert generated_labels[-1].shape[0] == required_num
        self.values = np.vstack(generated_signals)
        self.labels = np.vstack(generated_labels)[:, selected].squeeze()
    
        self.res_min = np.load(os.path.dirname(labels_path)+"/thirdparty/res_min.npy")
        self.res_max = np.load(os.path.dirname(labels_path)+"/thirdparty/res_max.npy")
        self.res_min = self.get_12_from_8(self.res_min)
        self.res_max = self.get_12_from_8(self.res_max)

        self.values = self.get_12_from_8(self.values)
        self.values = self.values*(self.res_max - self.res_min) + self.res_min

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.values[index]).float(), torch.tensor(self.labels[index], dtype=bool)
            
    @staticmethod
    # Obtain 12 leads from 8 leads
    def get_12_from_8(x):
        sec = x[:,0,:] + x[:,1,:]
        avr = -0.5 * (x[:,0,:] + sec)
        avl = 0.5 * (x[:,0,:] - x[:,1,:])
        avf = 0.5 * (sec + x[:,1,:])
        return np.stack([x[:,0,:], sec, x[:,1,:], avr, avl, avf, x[:,2,:], x[:,3,:], x[:,4,:], x[:,5,:], x[:,6,:], x[:,7,:]], axis=1)