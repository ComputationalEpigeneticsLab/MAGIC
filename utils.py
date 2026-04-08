
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import os
import pandas as pd
import glob
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from collections import defaultdict



class Prepare_Train_Datasets_FusionModel(Dataset):
    def __init__(self, fold, train_wsi_path):
        fold_path = os.path.join(train_wsi_path, f'fold_{fold}')
        csv_file = os.path.join(fold_path, 'train_data.csv')
        pd_data = pd.read_csv(csv_file)
        self.data_list = pd_data['filepath'].tolist()
        self.exp_path = os.path.join(fold_path, 'train_protein.csv')
        self.fold_path = fold_path

    def getpath(self):
        all_samples = self.data_list
        all_paths = []
        for sample in all_samples:
            data_paths = glob.glob(os.path.join(sample, "*", 'patch_features.csv'))
            all_paths.extend(data_paths)
        return all_paths


    def __getitem__(self, item):
        data_path = self.getpath()[item]
        data_df = pd.read_csv(data_path,index_col=0)
        sample = data_path.split('/')[-3]
        data_exp = pd.read_csv(self.exp_path, usecols = [sample])
        data_exp = data_exp.values
        data_exp = np.transpose(data_exp)
        data = torch.tensor(data_df.values, dtype=torch.float32)
        exp_data = torch.tensor(data_exp, dtype=torch.float32)
        hovernet_path = os.path.join(self.fold_path,'hovernet/train',data_path.split('/')[-4],data_path.split('/')[-3],data_path.split('/')[-2],data_path.split('/')[-1])
        hovernet_data = pd.read_csv(hovernet_path,index_col=0)
        hovernet = torch.tensor(hovernet_data.values, dtype=torch.float32)
        label_name = data_path.split('/')[-4]
        if label_name == 'resistant':
            label = 0
        elif label_name == 'response':
            label = 1
        else:
            raise ValueError(f"Unknown label: {label_name}")
        return data, exp_data, hovernet, label


    def __len__(self):
        return len(self.getpath())



class Prepare_Val_Datasets_FusionModel(Dataset):
    def __init__(self, fold, train_wsi_path):
        fold_path = os.path.join(train_wsi_path, f'fold_{fold}')
        csv_file = os.path.join(fold_path, 'val_data.csv')
        pd_data = pd.read_csv(csv_file)
        self.data_list = pd_data['filepath'].tolist()
        self.exp_path = os.path.join(fold_path, 'val_protein.csv')
        self.fold_path = fold_path

    def getpath(self):
        all_samples = self.data_list
        all_paths = []
        for sample in all_samples:
            data_paths = glob.glob(os.path.join(sample, "*", 'patch_features.csv'))
            all_paths.extend(data_paths)
        return all_paths


    def __getitem__(self, item):
        data_path = self.getpath()[item]
        data_df = pd.read_csv(data_path,index_col=0)
        sample = data_path.split('/')[-3]
        data_exp = pd.read_csv(self.exp_path, usecols = [sample])
        data_exp = data_exp.values
        data_exp = np.transpose(data_exp)
        data = torch.tensor(data_df.values, dtype=torch.float32)
        exp_data = torch.tensor(data_exp, dtype=torch.float32)
        hovernet_path = os.path.join(self.fold_path, 'hovernet/val', data_path.split('/')[-4],
                                     data_path.split('/')[-3], data_path.split('/')[-2], data_path.split('/')[-1])
        hovernet_data = pd.read_csv(hovernet_path, index_col=0)
        hovernet = torch.tensor(hovernet_data.values, dtype=torch.float32)
        label_name = data_path.split('/')[-4]

        if label_name == 'resistant':
            label = 0
        elif label_name == 'response':
            label = 1
        else:
            raise ValueError(f"Unknown label: {label_name}")
        return data, exp_data,hovernet, label, sample


    def __len__(self):
        return len(self.getpath())




class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=50):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            stop_epoch (int): Earliest epoch possible for stopping
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 100000

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class EarlyStopping_auc:
    def __init__(self, patience=10, stop_epoch=10):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved(20).
            stop_epoch (int): Earliest epoch possible for stopping(50)
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.auc_max = 0

    def __call__(self, epoch, y_true,y_score, model, ckpt_name = 'checkpoint.pt'):

        score = roc_auc_score(y_true, y_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, auc, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        print(f'Validation auc increased ({self.auc_max:.6f} --> {auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.auc_max = auc




def VAE_loss_function(x, x_recon, z_mu, z_logvar):
    # Reconstruction loss (Mean Squared Error)
    recon_loss = F.mse_loss(x_recon, x)

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

    # Total loss
    loss = recon_loss + kl_loss
    return loss


def cal_metrics(model, loader, device):
    model.eval()
    patient_true = defaultdict(list)
    patient_scores = defaultdict(list)
    patient_preds = defaultdict(list)
    with torch.no_grad():
        for i, (wsi_data, exp_data, hovernet, label, sample_id) in enumerate(loader):
            wsi_data, exp_data, hovernet, label = wsi_data.to(device), exp_data.to(device), hovernet.to(device),label.to(device)
            #
            label = label.float()
            outputs, _, _, _, _= model(wsi_data, hovernet, exp_data)
            outputs = outputs.squeeze()
            predicted = (outputs > 0.5).float()
            #
            patient_id = sample_id
            patient_true[patient_id].append(label.item())
            patient_scores[patient_id].append(outputs.item())
            patient_preds[patient_id].append(predicted.item())

    y_true, y_pred, y_score = [], [], []
    for patient_id in patient_true:
        #
        true_label = patient_true[patient_id][0]
        y_true.append(true_label)
        #
        avg_score = np.mean(patient_scores[patient_id])
        y_score.append(avg_score)
        #
        avg_pred = np.mean(patient_preds[patient_id])
        y_pred.append(round(avg_pred))

    #
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)
    return specificity, sensitivity, accuracy, precision, recall, f1, auc_roc, auc_pr, y_true, y_pred, y_score
