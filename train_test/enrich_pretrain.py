import sys
sys.path.append("../")

import torch
torch.set_default_dtype(torch.float32)
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle

from neuromodel import CNN1dTrainer
from ecg_ptbxl_benchmarking.code.models.xresnet1d import xresnet1d101
from data.data_modules import ECGDataset, GeneratedECGDataset
import os

import argparse
import json

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative models test")

    parser.add_argument('-origin_data_path', type=str, required=True)
    parser.add_argument('-generated_data_path', type=str, required=True)
    parser.add_argument('-model_type', type=str, required=True)
    parser.add_argument('-dataset_name', type=str, required=True)

    parser.add_argument('--res_path', type=str, default="./")
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=24)
    args = parser.parse_args()

    assert args.model_type in ["p2p", "wg*", "nvae"]

    with open(os.path.join(args.origin_data_path, "label2id.pickle"), "rb") as f:
        label2id = pickle.load(f)
    selected_classes = {"ptb-xl": ['426783006', '39732003', '164873001', '164889003', '427084000', '270492004', '426177001', '164934002'],
                        "georgia": ['426783006', '39732003', '164873001', '164889003', '427084000', '270492004', '426177001', '164934002'],
                        "ningbo": ['426783006', '39732003', '164873001', '427084000', '270492004', '426177001', '164934002']
    }

    ptbxl_train_ds = ECGDataset(args.origin_data_path, "ptb-xl", label2id, selected_classes["ptb-xl"], type="classify", option='train')
    ptbxl_val_ds = ECGDataset(args.origin_data_path, "ptb-xl", label2id, selected_classes["ptb-xl"], type="classify", option='val')
    ptbxl_test_ds = ECGDataset(args.origin_data_path, "ptb-xl", label2id, selected_classes["ptb-xl"], type="classify", option='test')
    
    orig_train_ds = ECGDataset(args.origin_data_path, args.dataset_name, label2id, selected_classes[args.dataset_name], type="classify", option='train')
    orig_val_ds = ECGDataset(args.origin_data_path, args.dataset_name, label2id, selected_classes[args.dataset_name], type="classify", option='val')
    orig_test_ds = ECGDataset(args.origin_data_path, args.dataset_name, label2id, selected_classes[args.dataset_name], type="classify", option='test')

    results = {}

    label2id_pretrain = {k:i for i, k in enumerate(selected_classes["ptb-xl"])}
    label2id_tune = {k:i for i, k in enumerate(selected_classes[args.dataset_name])}

    for i_step in np.arange(0.0, 1.1, 0.1):
        # Model pretraining on PTB-XL
        i = round(i_step, 1)
        print("Working with {}".format(round(i,1)))
        if i == 0:
            train_ds = ptbxl_train_ds
        else:
            gen_train_ds = GeneratedECGDataset(os.path.join(args.origin_data_path, "ptb-xl", "labels.npy"), args.generated_data_path, args.model_type, "imbalance", label2id_pretrain, selected_classes["ptb-xl"], i)
            train_ds = torch.utils.data.ConcatDataset([ptbxl_train_ds, gen_train_ds])
        model = xresnet1d101(input_channels=12, num_classes=len(label2id_pretrain))
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        trainer = CNN1dTrainer(args.model_type + '_pretrained_ptbxl_' + str(i), label2id_pretrain, model, opt, torch.nn.BCEWithLogitsLoss(), train_ds, ptbxl_val_ds, 
                            ptbxl_test_ds, args.res_path, cuda_id=args.device, return_model=True)
        model = trainer.train(args.num_epochs)
        # Model fine-tuning
        model[-1][-1] = nn.Linear(512, len(label2id_tune))
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        trainer = CNN1dTrainer(args.model_type + f'_fine_tuned_{args.dataset_name}' + str(i), label2id_tune, model, opt, torch.nn.BCEWithLogitsLoss(), orig_train_ds, orig_val_ds, 
                            orig_test_ds, args.res_path, cuda_id=args.device, return_model=False)
        test_res = trainer.train(args.num_epochs)
        results[i] = test_res

    with open(f"./results_fine_tuned_{args.dataset_name}_{args.model_type}.json", 'w') as f:
        json.dump(results, f)