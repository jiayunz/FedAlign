import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import torch
from torchnlp.encoders.text import StaticTokenizerEncoder
from torch.nn.utils.rnn import pad_sequence

from .utils import random_k_class, SingleLabelDataset

def collate_fn(batch):
    batch_data, batch_labels, batch_label_masks, batch_label_masks_true, batch_weight, batch_targets = zip(*batch)

    batch_data = pad_sequence([torch.LongTensor(x) for x in batch_data], batch_first=True, padding_value=0)
    batch_labels = torch.FloatTensor(batch_labels)
    batch_label_masks = torch.LongTensor(batch_label_masks)
    batch_label_masks_true = torch.LongTensor(batch_label_masks_true)
    batch_weight = torch.FloatTensor(batch_weight)
    batch_targets = torch.LongTensor(batch_targets)

    return batch_data, batch_labels, batch_label_masks, batch_label_masks_true, batch_weight, batch_targets


def load_data(data_dir, n_client, k_class=5, seed=4321):
    target_names = ['trade', 'grain', 'ship', 'acq', 'earn', 'money-fx', 'interest', 'crude']

    target_name_index = {tn: i for i, tn in enumerate(target_names)}

    train_data = pd.read_csv(os.path.join(data_dir, f'r8-train-stemmed.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, f'r8-test-stemmed.csv'))
    val_data = pd.read_csv(os.path.join(data_dir, f'r8-dev-stemmed.csv'))

    max_len = 1000
    encoder = StaticTokenizerEncoder(train_data['text'], tokenize=lambda s: s.split()[:max_len], min_occurrences=100)
    print('vocab_size', encoder.vocab_size)
    X_train_all = [encoder.encode(x).numpy() for x in train_data['text']]
    y_train_all = np.eye(len(target_names))[[target_name_index[l] for l in train_data['intent']]]

    label_count = np.sum(y_train_all, axis=0)

    X_train = []
    Y_train = []
    if n_client > 1:
        kf = KFold(n_splits=n_client, random_state=seed, shuffle=True)
        for c, (_, client_indices) in enumerate(kf.split(range(len(y_train_all)))):
            X_train.append([X_train_all[i] for i in client_indices])
            Y_train.append(y_train_all[client_indices])
    else:
        X_train = X_train_all
        Y_train = y_train_all

    text_length = []
    for client_data in X_train:
        text_length.extend([len(x) for x in client_data])
    print('text length', np.average(text_length))

    X_test = [encoder.encode(x).numpy() for x in test_data['text']]
    Y_test = np.eye(len(target_names))[[target_name_index[l] for l in test_data['intent']]]
    X_val = [encoder.encode(x).numpy() for x in val_data['text']]
    Y_val = np.eye(len(target_names))[[target_name_index[l] for l in val_data['intent']]]

    M_train, M_train_orig, client2labels = random_k_class(Y_train, n_client, k_class)

    label_count += np.sum(Y_val, axis=0)
    label_count += np.sum(Y_test, axis=0)
    imbalance_factor = np.min(label_count) / np.max(label_count)
    print('total imbalance', imbalance_factor)
    print('loaded X and Y')

    print(client2labels)

    testData = SingleLabelDataset(X_test, Y_test, np.zeros_like(Y_test), M_true=np.zeros_like(Y_test), active_targets=range(len(target_names)), target_names=target_names, vocab=encoder.vocab)
    valData = SingleLabelDataset(X_val, Y_val, np.zeros_like(Y_val), M_true=np.zeros_like(Y_val), active_targets=range(len(target_names)), target_names=target_names, vocab=encoder.vocab)

    trainData = []
    for c in range(n_client):
        trainData.append(SingleLabelDataset(X_train[c], Y_train[c], M_train[c], M_true=M_train_orig[c], active_targets=[l for l in client2labels[c]], target_names=target_names, vocab=encoder.vocab))

    return trainData, valData, testData