import numpy as np
import torch
torch.set_default_dtype(torch.float32)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score
from utils import find_threshold

import os
from tqdm import tqdm
import pickle
    
    
##### Trainer for 1d-CNN model #####
class CNN1dTrainer:
    def __init__(self, class_name, label2id,
                 model, optimizer, loss,
                 train_dataset, val_dataset, test_dataset, model_path,
                 batch_size=128, cuda_id=1, return_model=False):
        
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.return_model = return_model
        self.id2label = {v:k for k,v in label2id.items()}
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.result_output = {}

        self.batch_size = batch_size

        self.device = torch.device("cuda:" + str(cuda_id) if (torch.cuda.is_available() or cuda_id != -1) else "cpu")
        self.model = self.model.to(self.device)

        self.global_step = 0
        self.alpha = 0.8
        
        self.class_name = class_name
        
        self.result_output['class'] = class_name
        
        os.makedirs(model_path + "/models" + "/" +self.class_name, exist_ok=True)
        os.makedirs(model_path + "/summary" + "/" + self.class_name, exist_ok=True)
        self.writer = SummaryWriter(model_path + "/summary" + "/" + self.class_name)    
        self.model_path = model_path 

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self, num_epochs):
        
        model = self.model
        optimizer = self.optimizer
        
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True, batch_size=self.batch_size, num_workers=8)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True, batch_size=len(self.val_dataset), num_workers=8)
        
        best_val, best_test = -38, -13
        for epoch in tqdm(range(num_epochs)):
            model.train()
            train_logits = []
            train_gts = []
            torch.manual_seed(epoch)
            for batch in self.train_loader:
                image, label = batch
                image = image.to(self.device)
                label = label.to(self.device)

                optimizer.zero_grad()
                logits = model(image).squeeze()
                train_logits.append(logits.cpu().detach())
                train_gts.append(label.cpu())
                loss = self.loss(logits, label.float())
                loss.backward()
                optimizer.step()
                self.writer.add_scalar("Train Loss", loss.item(), global_step=self.global_step)
                self.global_step += 1

            train_gts = np.concatenate(train_gts)
            train_logits = train_logits
            train_gts = train_gts

            preds = torch.sigmoid(torch.cat(train_logits)).numpy()
            res_metric = []

            if len(train_gts.shape) == 1:
                train_gts = np.expand_dims(train_gts, -1)
                preds = np.expand_dims(preds, -1)
            for i in range(train_gts.shape[1]):
                res_metric.append(roc_auc_score(train_gts[:,i], preds[:,i]))
            self.writer.add_scalar("Train AP/{}".format(self.class_name), np.mean(res_metric), global_step=epoch)

            val_logits = []
            val_gts = []
            model.eval()
            with torch.no_grad():
                for batch in self.val_loader:
                    image, label = batch
                    image = image.to(self.device)
                    label = label.to(self.device)
                    logits = model(image).cpu()
                    gts = label.cpu()
                    val_logits.append(logits)
                    val_gts.append(gts)

                gts = np.concatenate(val_gts)

                preds = torch.sigmoid(torch.cat(val_logits)).numpy()
                res_metric = []
                if len(gts.shape) == 1:
                    gts = np.expand_dims(gts, -1)
                for i in range(gts.shape[1]):
                    res_metric.append(roc_auc_score(gts[:,i], preds[:,i]))
                mean_val = np.mean(res_metric)
                if mean_val > best_val:
                    self.save_checkpoint(self.model_path + "/models" + "/" +self.class_name+"/best_checkpoint.pth")
                    best_val = mean_val
                    self.result_output['threshold'] = find_threshold(gts, logits)
                    
                    best_test = self.test(self.model, self.test_dataset, epoch)
                    self.writer.add_scalar("Val AP/{}".format(self.class_name), mean_val, global_step=epoch)
                else:
                    best_test = self.test(self.model, self.test_dataset, epoch)
                    self.writer.add_scalar("Val AP/{}".format(self.class_name), mean_val, global_step=epoch)

        self.writer.add_text("Final test metric/{}".format(self.class_name), str(round(np.mean(list(best_test.values())), 4)))
        with open(self.model_path + "/models" + "/" +self.class_name+"/log.pickle", 'wb') as handle:
            pickle.dump(self.result_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.return_model:
            return self.model
        else:
            return best_test

       
    def test(self, model, test_dataset, epoch):
        model.eval()
        
        test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=self.batch_size, num_workers=8)
        test_logits = []
        test_gts = []
        for batch in test_loader:
            image, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                logits = model(image).cpu()
                gts = label.cpu()
                test_logits.append(logits)
                test_gts.append(gts)

        preds = torch.sigmoid(torch.cat(test_logits)).numpy()
        gts = np.concatenate(test_gts)

        res_metric = []
        res_dict = {}
        if len(gts.shape) == 1:
            gts = np.expand_dims(gts, -1)

        for i in range(gts.shape[1]):
            metric_val = roc_auc_score(gts[:,i], preds[:,i])
            res_metric.append(metric_val)
            res_dict[self.id2label[i]] = metric_val
        self.writer.add_scalar("Test ROC-AUC/{}".format(self.class_name), np.mean(res_metric), global_step=epoch)
        return res_dict