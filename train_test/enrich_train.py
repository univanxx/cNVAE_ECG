import sys
sys.path.append("../")

import torch
torch.set_default_dtype(torch.float32)
import torch.optim as optim
import numpy as np

from neuromodel import CNN1dTrainer
from ecg_ptbxl_benchmarking.code.models.xresnet1d import xresnet1d101
from data.data_modules import ECGDataset, GeneratedECGDataset
import os
import pickle
import argparse
import json

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative models test")

    parser.add_argument('-origin_data_path', type=str, required=True)
    parser.add_argument('-generated_data_path', type=str, required=True)
    parser.add_argument('-model_type', type=str, required=True)
    parser.add_argument('-task_type', type=str, required=True)

    parser.add_argument('--res_path', type=str, default="./")
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=24)
    args = parser.parse_args()

    assert args.model_type in ["p2p", "wg*", "nvae"]

    # Classes to work with
    selected_classes = ["164865005"]
    with open(os.path.join(args.origin_data_path, "label2id.pickle"), "rb") as f:
        label2id = pickle.load(f)

    orig_train_ds = ECGDataset(args.origin_data_path, "ptb-xl", label2id, selected_classes, type="classify", option='train')
    orig_val_ds = ECGDataset(args.origin_data_path, "ptb-xl", label2id, selected_classes, type="classify", option='val')
    orig_test_ds = ECGDataset(args.origin_data_path, "ptb-xl", label2id, selected_classes, type="classify", option='test')
    
    label2id = {k:i for i, k in enumerate(selected_classes)}
    results = {}
    for i_step in np.arange(0.0, 1.1, 0.1):
        i = round(i_step, 1)
        print("Working with {}".format(round(i,1)))
        if i == 0:
            # Without additional generated data
            train_ds = orig_train_ds
        else:
            gen_train_ds = GeneratedECGDataset(os.path.join(args.origin_data_path, "ptb-xl", "labels.npy"), args.generated_data_path, args.model_type, args.task_type, label2id, selected_classes, i)
            train_ds = torch.utils.data.ConcatDataset([orig_train_ds, gen_train_ds])
        # Model training on PTB-XL   
        model = xresnet1d101(input_channels=12, num_classes=len(selected_classes))
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        trainer = CNN1dTrainer(args.model_type + '_' + str(i), label2id, model, opt, torch.nn.BCEWithLogitsLoss(), train_ds, orig_val_ds, 
                            orig_test_ds, args.res_path, cuda_id=args.device)
        test_res = trainer.train(args.num_epochs)
        results[i] = test_res

    with open(f"./results_ptbxl_{args.task_type}_{args.model_type}.json", 'w') as f:
        json.dump(results, f)