import os

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd

from torch.nn.utils.rnn import pad_sequence
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor

from .utils import partial_labeling_by_category

def collate_fn(batch):
    batch_data, batch_labels, batch_label_masks, batch_label_masks_true, batch_weight, batch_targets = zip(*batch)

    batch_data = pad_sequence([torch.LongTensor(x) for x in batch_data], batch_first=True, padding_value=0)
    batch_labels = torch.FloatTensor(batch_labels)
    batch_label_masks = torch.LongTensor(batch_label_masks)
    batch_label_masks_true = torch.LongTensor(batch_label_masks_true)
    batch_weight = torch.FloatTensor(batch_weight)
    batch_targets = torch.LongTensor(batch_targets)

    return batch_data, batch_labels, batch_label_masks, batch_label_masks_true, batch_weight, batch_targets



def load_data(data_path, n_client=1, seed=4321):
    def load_csv_to_array(csv_path, code2idx):
        csv_data = pd.read_csv(csv_path, dtype={'LABELS': str})

        X = csv_data['TEXT'].to_numpy()
        Y = np.zeros((len(X), len(code2idx)))
        for i, labels in enumerate(csv_data['LABELS'].to_numpy()):
            pos_labels = [code2idx[j] for j in str(labels).split(';') if j in code2idx]
            Y[i][pos_labels] = 1

        return X, Y

    icd2names = pd.read_csv(os.path.join(data_path, 'ICD9_descriptions'), sep='\\t', header=0).set_index('@')
    code2idx = {code: i for i, code in enumerate(icd2names.index)}

    X_train, Y_train = load_csv_to_array(os.path.join(data_path, 'mimic3/train_50.csv'), code2idx)
    print('average positive labels per patient:', np.average(Y_train.sum(-1)))
    X_val, Y_val = load_csv_to_array(os.path.join(data_path, 'mimic3/dev_50.csv'), code2idx)
    X_test, Y_test = load_csv_to_array(os.path.join(data_path, 'mimic3/test_50.csv'), code2idx)

    stopwords = ['admission', 'discharge', 'date', 'birth']
    max_len = 1000
    encoder = StaticTokenizerEncoder(X_train, tokenize=lambda s: [w for w in s.split() if w not in stopwords][:max_len], min_occurrences=100)
    X_train = [encoder.encode(x).numpy() for x in X_train]
    X_val = [encoder.encode(x).numpy() for x in X_val]
    X_test = [encoder.encode(x).numpy() for x in X_test]

    # remove classes with less than samples
    Y_all = np.concatenate([Y_train, Y_val, Y_test])
    preserve_idx = []
    for code, i in code2idx.items():
        if (Y_all[:, i] == 1).sum() > 10:
            preserve_idx.append(i)

    Y_all = Y_all[:, preserve_idx]
    Y_train = Y_train[:, preserve_idx]
    Y_test = Y_test[:, preserve_idx]
    Y_val = Y_val[:, preserve_idx]
    code2idx = {code: i for code, i in code2idx.items() if i in preserve_idx}
    idx2code = [code for code, i in code2idx.items()]

    target_names =icd2names['ICD9 Hierarchy Root'].loc[idx2code].to_numpy()
    label_count = np.sum(Y_all, axis=0)
    imbalance_factor = np.min(label_count) / np.max(label_count)
    print('label_count.shape', label_count.shape, 'imbalance', imbalance_factor)

    targetname2category = {}
    for code in code2idx:
        name = icd2names['ICD9 Hierarchy Root'].loc[code]
        if code.startswith('E') or code.startswith('V'):
            targetname2category[name] = 'external causes of injury and supplemental classification'
        elif float(code) < 140:
            targetname2category[name] = 'infectious and parasitic diseases'
        elif float(code) < 240:
            targetname2category[name] = 'neoplasms'
        elif float(code) < 280:
            targetname2category[name] = 'endocrine, nutritional and metabolic diseases, and immunity disorders'
        elif float(code) < 290:
            targetname2category[name] = 'diseases of the blood and blood-forming organs'
        elif float(code) < 320:
            targetname2category[name] = 'mental disorders'
        elif float(code) < 390:
            targetname2category[name] = 'diseases of the nervous system and sense organs'
        elif float(code) < 460:
            targetname2category[name] = 'diseases of the circulatory system'
        elif float(code) < 520:
            targetname2category[name] = 'diseases of the respiratory system'
        elif float(code) < 580:
            targetname2category[name] = 'diseases of the digestive system'
        elif float(code) < 630:
            targetname2category[name] = 'diseases of the genitourinary system'
        elif float(code) < 680:
            targetname2category[name] = 'complications of pregnancy, childbirth, and the puerperium'
        elif float(code) < 710:
            targetname2category[name] = 'diseases of the skin and subcutaneous tissue'
        elif float(code) < 740:
            targetname2category[name] = 'diseases of the musculoskeletal system and connective tissue'
        elif float(code) < 760:
            targetname2category[name] = 'congenital anomalies'
        elif float(code) < 780:
            targetname2category[name] = 'certain conditions originating in the perinatal period'
        elif float(code) < 800:
            targetname2category[name] = 'symptoms, signs, and ill-defined conditions'
        elif float(code) < 1000:
            targetname2category[name] = 'injury and poisoning'
        else:
            raise ValueError('ICD Code Error')
    category2targetnames = defaultdict(list)
    for tn in target_names:
        category2targetnames[targetname2category[tn]].append(tn)
    print('loaded X and Y')

    if n_client > 1:
        X_train, Y_train, M_train, M_train_orig, client2labels = partial_labeling_by_category(X_train, Y_train, np.zeros_like(Y_train), n_client, targetname2category, target_names, seed=seed)
        X_val, Y_val, M_val, M_val_orig, client2labels = partial_labeling_by_category(X_val, Y_val, np.zeros_like(Y_val), n_client, targetname2category, target_names, seed=seed)
    else:
        X_train, Y_train, M_train, M_train_orig = [X_train], [Y_train], [np.zeros_like(Y_train)], [np.zeros_like(Y_train)]
        X_val, Y_val, M_val, M_val_orig = [X_val], [Y_val], [np.zeros_like(Y_val)], [np.zeros_like(Y_val)]
        client2labels = {0: range(len(M_train[0][0]))}

    print(client2labels)
    trainData = []

    testData = MyDataset(X_test, Y_test, np.zeros_like(Y_test), M_true=np.zeros_like(Y_test), active_targets=range(len(Y_test[0])), target_names=target_names, vocab=encoder.vocab)

    X_val_concat = []
    Y_val_concat = []
    M_val_concat = []
    M_val_orig_concat = []

    for c in range(n_client):
        X_val_concat.extend(X_val[c])
        Y_val_concat.extend(Y_val[c])
        M_val_concat.extend(M_val[c])
        M_val_orig_concat.extend(M_val_orig[c])

        trainData.append(MyDataset(X_train[c], Y_train[c], M_train[c], M_true=M_train_orig[c], active_targets=[l for l in client2labels[c]], target_names=target_names, vocab=encoder.vocab))

    valData = MyDataset(X_val_concat, Y_val_concat, M_val_concat, M_true=M_val_orig_concat, active_targets=range(len(Y_val_concat[0])), target_names=target_names, vocab=encoder.vocab)

    return trainData, valData, testData


class MyDataset(Dataset):
    def __init__(self, X, Y, M, M_true, active_targets=None, target_names=None, vocab=None):
        self.data = X
        self.labels = np.array(Y)
        self.label_masks = np.array(M)
        self.label_masks_true = np.array(M_true)
        self.target_names = target_names
        self.vocab = vocab

        if active_targets is None:
            self.active_targets = np.where(np.any(self.label_masks == 0, dim=0))[0]
        else:
            self.active_targets = active_targets

        self.get_instance_weights()
        self.data_len = np.array([len(x) for x in self.data])


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        label_masks = self.label_masks[idx]
        label_masks_true = self.label_masks_true[idx]
        instance_weights = self.instance_weights[idx]
        return data, labels, label_masks, label_masks_true, instance_weights, self.active_targets

    def get_instance_weights(self):
        n_classes = len(self.labels[0])
        # Count each class frequency (pos/neg) for each label
        pos_count = np.ones((n_classes))  # avoid nan
        neg_count = np.ones((n_classes))
        for example_y, example_m in zip(self.labels, self.label_masks):
            for i, (y, m) in enumerate(zip(example_y, example_m)):
                if m == 1:
                    continue
                if y == 1:
                    pos_count[i] += 1
                elif y == 0:
                    neg_count[i] += 1
        self.num_samples = pos_count - 1
        self.pos_weight = neg_count / (pos_count + neg_count)
        self.neg_weight = pos_count / (pos_count + neg_count)

        self.instance_weights = []
        for y, m in zip(self.labels, self.label_masks):
            weight = (y * self.pos_weight + (1 - y) * self.neg_weight) * (1 - m)
            self.instance_weights.append(weight)
