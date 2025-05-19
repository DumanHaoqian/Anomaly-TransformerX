from data_loader import get_loader_segment, SMDSegLoader
import torch

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timezone, timedelta
from sklearn.metrics import silhouette_score

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold



class BeijingFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, timezone.utc)
        return dt.astimezone(timezone(timedelta(hours=8)))
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="test_svdd_with_grid_search.log",
)
for handler in logging.getLogger().handlers:
    handler.setFormatter(BeijingFormatter(handler.formatter._fmt, handler.formatter.datefmt))

#####logging setting

fusion_test = np.load('fusion_test.npy')
fusion_train = np.load('fusion_train.npy')

test_loader = get_loader_segment('./SMD', batch_size=32 ,win_size=1, step=1, mode='test', dataset='SMD')
y_test_list = []
for i, (input_data, labels) in enumerate(test_loader):

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    y_test_list.append(labels)

y_test = np.concatenate(y_test_list, axis=0)

logging.info(f"fusion_train shape:{fusion_train.shape}" )
logging.info(f"fusion_test shape:{fusion_test.shape}" )

x_train = fusion_train
x_test = fusion_test



import BaseSVDD
# SVDD algorithm from https://github.com/iqiukp/SVDD-Python/, there is no parameter for grid search

svdd = BaseSVDD.BaseSVDD()


sample_size = 10000  # dateset size for 708405 is too large, make it smaller
indices = np.random.choice(len(x_train), size=sample_size, replace=False)
x_train_sub = x_train[indices]

sample_size = 10000  # dateset size for 708420 is too large, make it smaller
indices = np.random.choice(len(x_test), size=sample_size, replace=False)
x_test_sub = x_test[indices]
y_test_sub = y_test[indices]
# SVDD use -1 for negeative samples and 1 for positive samples
y_test_sub[y_test_sub == 0] = -1

def custom_scorer(estimator, X):
    labels = estimator.predict(X)
    labels = np.where(labels == -1, 0, labels) 
    return silhouette_score(X, labels.ravel())


best_score = 0 
best_params = {}

iteration = 0

C_list = [1.5]
gamma_list =['auto']
kernel_list = ['rbf']
njob = 4



for C in C_list:
    for gamma in gamma_list:
        for kernel in kernel_list:
            logging.info(f"Iteration for base SVDD: {iteration} begin")
            try:
                fold_scores = []  
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                for train_idx, val_idx in kfold.split(x_train_sub):
 
                    X_fold_train = x_train_sub[train_idx]
                    X_fold_val = x_train_sub[val_idx]
                    
                    svdd = BaseSVDD.BaseSVDD(C=C, kernel=kernel, gamma=gamma, n_jobs=njob)
                    svdd.fit(X_fold_train) 
                    fold_score = custom_scorer(svdd, X_fold_val)
                    fold_scores.append(fold_score)
                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'C': C, 'kernel': kernel, 'gamma': gamma}
                iteration += 1
                logging.info(f"Iteration for base SVDD: {iteration}, C: {C}, kernel: {kernel}, gamma: {gamma}, Score: {mean_score}")
                logging.info(f"Best parameters now: {str(best_params)}")
                logging.info(f"Best score now: {str(best_score)}")

            except Exception as e:
                logging.error(f"Error in iteration {iteration}: {e}")
                logging.info(f"Iteration for base SVDD: {iteration}, C: {C}, kernel: {kernel}, gamma: {gamma}")
                continue
        

svdd = BaseSVDD.BaseSVDD(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'], n_jobs=njob)
svdd.fit(x_train_sub)
y_test_predict = svdd.predict(X = x_test_sub)

# get negative samples back to 0
y_test_predict = np.where(y_test_predict == -1, 0, y_test_predict)
y_test_sub = np.where(y_test_sub == -1, 0, y_test_sub)
svdd_precision = precision_score(y_test_sub, y_test_predict)
svdd_recall = recall_score(y_test_sub, y_test_predict)
svdd_f1 = f1_score(y_test_sub, y_test_predict)

logging.info("Base SVDD Precision:")
logging.info(f"Precision: {svdd_precision}")
logging.info(f"Recall: {svdd_recall}")
logging.info(f"F1 Score: {svdd_f1}")
logging.info(f"Best parameters: {best_params}")

radius = svdd.radius
indices = np.random.choice(len(x_test_sub), size=500, replace=False)
x_test_sub_for_plot = x_test_sub[indices]
distance = svdd.get_distance(x_test_sub_for_plot)
svdd.plot_distance(radius, distance)


################# DEEP SVDD
logging.info("go deep")


from pyod.models.deep_svdd import DeepSVDD

hidden_neurons_list = [[64, 32], [128, 64], [256, 128]]
hidden_activation_list = ['relu', 'tanh', 'leaky_relu']
dropout_list = [0.1, 0.2, 0.3]

y_test = y_test.reshape(-1) 

best_score = 0
best_params = {}

iteration = 0


for hidden_neurons in hidden_neurons_list:
    for hidden_activation in hidden_activation_list:
        for dropout in dropout_list:
            logging.info(f"Iteration for Deep SVDD: {iteration} begin")
            fold_scores = []  
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in kfold.split(x_train_sub):
                X_fold_train = x_train_sub[train_idx]
                X_fold_val = x_train_sub[val_idx]
                try:
                    deep_svdd = DeepSVDD(n_features= 114, hidden_neurons=hidden_neurons, hidden_activation=hidden_activation, 
                                     dropout_rate = dropout, batch_size = 32, epochs = 50)
                    deep_svdd.fit(X_fold_train)
                    fold_score = custom_scorer(deep_svdd, X_fold_val)
                    fold_scores.append(fold_score)
                except Exception as e:
                    logging.error(f"Error in iteration {iteration}: {e}")
                    logging.info(f"Iteration for deep SVDD: {iteration}, hidden_neurons: {hidden_neurons}, hidden_activation: {hidden_activation}, dropout: {dropout}")
                    continue
            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'hidden_neurons': hidden_neurons, 'hidden_activation': hidden_activation, 'dropout': dropout}
            iteration += 1
            logging.info(f"Iteration for deep SVDD: {iteration}, hidden_neurons: {hidden_neurons}, hidden_activation: {hidden_activation}, dropout: {dropout}, Score: {mean_score}")
            logging.info(f"Best parameters now: {str(best_params)}")


deep_svdd = DeepSVDD(n_features= 114, hidden_neurons=best_params['hidden_neurons'], 
                             hidden_activation=best_params['hidden_activation'], dropout_rate = best_params['dropout'], 
                             batch_size = 32, epochs = 50)

deep_svdd.fit(x_train)

y_test_predict = deep_svdd.predict(x_test)

deep_svdd_precision = precision_score(y_test, y_test_predict)
deep_svdd_recall = recall_score(y_test, y_test_predict)
deep_svdd_f1 = f1_score(y_test, y_test_predict)
logging.info("Deep SVDD Precision:")
logging.info(f"Precision: {deep_svdd_precision}")
logging.info(f"Recall: {deep_svdd_recall}")
logging.info(f"F1 Score: {deep_svdd_f1}")
logging.info(f"Best parameters: {best_params}")


