import numpy as np
import pickle

import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

from .utils import random_k_class, SingleLabelDataset

def collate_fn(batch):
    batch_data, batch_labels, batch_label_masks, batch_label_masks_true, batch_weight, batch_targets = zip(*batch)

    batch_data = pad_sequence([torch.FloatTensor(x) for x in batch_data], batch_first=True, padding_value=0)
    batch_labels = torch.FloatTensor(batch_labels)
    batch_label_masks = torch.LongTensor(batch_label_masks)
    batch_label_masks_true = torch.LongTensor(batch_label_masks_true)
    batch_weight = torch.FloatTensor(batch_weight)
    batch_targets = torch.LongTensor(batch_targets)

    return batch_data, batch_labels, batch_label_masks, batch_label_masks_true, batch_weight, batch_targets


def load_data(data_path, label_path, k_class, seed=4321):
    X = pickle.load(open(data_path, 'rb'))
    Y = pickle.load(open(label_path, 'rb'))
    Y = [np.eye(12)[y] for y in Y]

    label_count = np.sum(np.concatenate(Y, axis=0), axis=0)
    imbalance_factor = np.max(label_count) / np.min(label_count)
    print('total imbalance', imbalance_factor)

    n_client = len(Y)
    M, M_orig, client2labels = random_k_class(Y, n_client, k_class)
    target_names = [
        'lying',
        'sitting',
        'standing',
        'walking',
        'running',
        'cycling',
        'nordic-walking',
        'ascending-stairs',
        'descending-stairs',
        'vacuum-cleaning',
        'ironing',
        'rope-jumping'
    ]

    print('loaded X and Y')

    print(client2labels)
    trainData = []

    X_val = []
    Y_val = []
    M_val = []
    M_val_orig = []

    X_test = []
    Y_test = []
    M_test = []
    M_test_orig = []

    for c in range(n_client):
        X_train_c, X_test_c, Y_train_c, Y_test_c, M_train_c, M_test_c, M_train_orig_c, M_test_orig_c = train_test_split(X[c], Y[c], M[c], M_orig[c], test_size=0.2, random_state=seed)
        X_train_c, X_val_c, Y_train_c, Y_val_c, M_train_c, M_val_c, M_train_orig_c, M_val_orig_c = train_test_split(X_train_c, Y_train_c, M_train_c, M_train_orig_c, test_size=0.25, random_state=seed)

        X_test.extend(X_test_c)
        Y_test.extend(Y_test_c)
        M_test.extend(M_test_c)
        M_test_orig.extend(M_test_orig_c)

        X_val.extend(X_val_c)
        Y_val.extend(Y_val_c)
        M_val.extend(M_val_c)
        M_val_orig.extend(M_val_orig_c)

        trainData.append(SingleLabelDataset(X_train_c, Y_train_c, M_train_c, M_true=M_train_orig_c, active_targets=[l for l in client2labels[c]], target_names=target_names))

    valData = SingleLabelDataset(np.array(X_val), np.array(Y_val), np.array(M_val), M_true=np.array(M_val_orig), active_targets=range(len(target_names)), target_names=target_names)
    testData = SingleLabelDataset(np.array(X_test), np.array(Y_test), np.array(M_test_orig), M_true=np.array(M_test_orig), active_targets=range(len(target_names)), target_names=target_names)

    label_count = np.sum(Y_test, axis=0)
    imbalance_factor = np.max(label_count) / np.min(label_count)
    print('test imbalance', imbalance_factor)

    return trainData, valData, testData
