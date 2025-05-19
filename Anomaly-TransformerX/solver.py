import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
import matplotlib.pyplot as plt
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,roc_auc_score
from sklearn.decomposition import PCA
def standardize(data, mean, std):
    return (data - mean) / std

def evaluate_anomalies(alpha,beta,rec_data, test_data, test_labels, threshold=0.95):
#使用rec_data去重构test_data数据，请将这两个图可视化，并保存到本地
        diff = np.abs(rec_data - test_data)
        #np.save("/home/haoqian/anomaly/Anomaly-TransformerX/rec/rec_data.npy",rec_data)
        #np.save("/home/haoqian/anomaly/Anomaly-TransformerX/rec/test_data.npy",test_data)
        anomalies = np.max(diff, axis=1) > threshold

        # 计算性能指标
        precision = precision_score(test_labels, anomalies)
        recall = recall_score(test_labels, anomalies)
        f1 = f1_score(test_labels, anomalies)
        accuracy = accuracy_score(test_labels, anomalies)
        auc_roc = roc_auc_score(test_labels, anomalies)
        # 打印结果
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC ROC Score:{auc_roc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        measurement=[precision,recall,f1,auc_roc,accuracy]
    
        return anomalies,measurement
def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=100, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                             mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                            mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                            mode='test', dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3, alpha=self.alpha,beta=self.beta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()
        total_loss = []
        
        with torch.no_grad():
            for i, (input_data, target_data) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                target = target_data.float().to(self.device)
                
                output, _ = self.model(input)
                loss = self.criterion(output, target)
                total_loss.append(loss.item())

        return np.average(total_loss)

    def train(self):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=100, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss_list = []
            epoch_time = time.time()
            self.model.train()

            for i, (input_data, target_data) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1

                input = input_data.float().to(self.device)
                target = target_data.float().to(self.device)

                output, _ = self.model(input)
                loss = self.criterion(output, target)
                
                loss_list.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss_list)
            vali_loss = self.vali(self.vali_loader)
            
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()

        print("======================TEST MODE======================")

        criterion = nn.MSELoss()
        total_loss = []
        all_outputs=[]

        with torch.no_grad():
            for i, (input_data, target_data,_) in enumerate(self.test_loader):
                input = input_data.float().to(self.device)
                target = target_data.float().to(self.device)

                output, _ = self.model(input)
                loss = criterion(output, target)
                total_loss.append(loss.item())
                all_outputs.append(output.cpu().numpy())
        all_outputs=np.concatenate(all_outputs,axis=0)
        all_outputs = all_outputs.reshape(-1, 38)  # 保持顺序，展开为 (708400, 38)
        #np.save(os.path.join(self.model_save_path, 'test_predictions.npy'), all_outputs)
        test_data=np.load("/home/haoqian/anomaly/Anomaly-TransformerX/dataset/SMD/SMD_test.npy")[:-20]
        test_labels=np.load("/home/haoqian/anomaly/Anomaly-TransformerX/dataset/SMD/SMD_test_label.npy")[:-20]



        anomalies,measurement = evaluate_anomalies(self.alpha,self.beta,all_outputs, test_data, test_labels)
        print("Combined Output Shape:", all_outputs.shape)
        
                
        test_loss = np.average(total_loss)
        print(f"Test Loss: {test_loss:.7f}")


        return test_loss,measurement