import sys
sys.path.append("../")

import numpy as np
import torch
torch.set_default_dtype(torch.float32)
import torch.optim as optim

from neuromodel import CNN1dTrainer
from ecg_ptbxl_benchmarking.code.models.xresnet1d import xresnet1d101
from data.PTBXLToDataset import CVConditional, CVGenerated, CVGeneratedGans

import argparse
import json

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generative model test")
    parser.add_argument('-model_name', type=str, required=True)
    parser.add_argument('-data_path', type=str, required=True)
    parser.add_argument('-generated_path', type=str, required=True)
    parser.add_argument('-fold', type=int, required=True)
    parser.add_argument('-model_type', type=str, required=True)

    parser.add_argument('--res_path', type=str, default="./")
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=24)
    args = parser.parse_args()

    assert args.model_type in ["p2p", "w_star", "nvae"]

    orig_train_ds = CVConditional("MI", 2560 // 10, args.fold, args.data_path, type="classify", 
                          option='train')
    orig_val_ds = CVConditional("MI", 2560 // 10, args.fold, args.data_path, type="classify", 
                          option='val')
    orig_test_ds = CVConditional("MI", 2560 // 10, args.fold, args.data_path, type="classify", 
                          option='test')
    
    results = {}

    for i_step in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]:
        i = round(i_step, 2)
        print("Working with {}".format(round(i,1)))
        if i == 0:
            train_ds = orig_train_ds
        else:
            if args.model_type == "nvae":
                gen_train_ds = CVGenerated("MI", args.model_name, 2560 // 10, args.fold, args.data_path, args.generated_path, 
                        option='proportion', proportion=i, seed=args.seed)
            else:
                gen_train_ds = CVGeneratedGans("MI", 2560 // 10, args.fold, args.data_path, 
                        args.generated_path, args.model_type, option='proportion', proportion=i, seed=args.seed)
            train_ds = torch.utils.data.ConcatDataset([orig_train_ds, gen_train_ds])

        torch.manual_seed(args.seed)
        model = xresnet1d101(input_channels=12, num_classes=1)
        opt = optim.AdamW(model.parameters(), lr=1.23e-02, weight_decay=1e-2)
        trainer = CNN1dTrainer(args.model_name + '_' + str(round(i,1)), model, opt, torch.nn.BCEWithLogitsLoss(), train_ds, orig_val_ds, 
                            orig_test_ds, args.res_path, cuda_id=args.device)
        test_res = trainer.train(args.num_epochs)
        results[i] = test_res
        print(test_res)
        if i == 0:
            print("Working with gan test")
            if args.model_type == "nvae":
                gen_test_ds = CVGenerated("MI", args.model_name, 2560 // 10, args.fold, args.data_path, args.generated_path, 
                        option='test', seed=args.seed)
            else:
                gen_test_ds = CVGeneratedGans("MI", 2560 // 10, args.fold, args.data_path, 
                        args.generated_path, args.model_type, option='test', seed=args.seed)
            model = xresnet1d101(input_channels=12, num_classes=1)
            opt = optim.AdamW(model.parameters(), lr=1.23e-02, weight_decay=1e-2)
            trainer = CNN1dTrainer(args.model_name+"_gan_test", model, opt, torch.nn.BCEWithLogitsLoss(), train_ds, orig_val_ds, 
                                gen_test_ds, args.res_path, cuda_id=args.device)
            test_res = trainer.train(args.num_epochs)
            results["gan_test"] = test_res
        elif i == 1.0:
            print("Working with gan train")
            train_ds = gen_train_ds
            if args.model_type == "nvae":
                gen_val_ds = CVGenerated("MI", args.model_name, 2560 // 10, args.fold, args.data_path, args.generated_path, 
                        option='val', seed=args.seed)
            else:
                gen_val_ds = CVGeneratedGans("MI", 2560 // 10, args.fold, args.data_path, 
                        args.generated_path, args.model_type, option='val', seed=args.seed)
            model = xresnet1d101(input_channels=12, num_classes=1)
            opt = optim.AdamW(model.parameters(), lr=1.23e-02, weight_decay=1e-2)
            trainer = CNN1dTrainer(args.model_name+"_gan_train", model, opt, torch.nn.BCEWithLogitsLoss(), train_ds, gen_val_ds, 
                                orig_test_ds, args.res_path, cuda_id=args.device)
            test_res = trainer.train(args.num_epochs)
            results["gan_train"] = test_res
        else:
            pass

        with open(f"./results_{args.model_type}_{args.model_name}_fold_{args.fold}_seed_{args.seed}.json", 'w') as f:
            json.dump(results, f)