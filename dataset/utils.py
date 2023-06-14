import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
import networkx as nx
import pandas as pd
from torch.utils.data import Dataset
from node2vec import Node2Vec

def partial_labeling_by_category(X, Y, M, n_client, targetname2category, target_names, seed):
    targetname2idx = {tn: i for i, tn in enumerate(target_names)}
    category2label = defaultdict(list)
    for tn in target_names:
        category2label[targetname2category[tn]].append(targetname2idx[tn])
    print(f'{len(category2label)} categories in targetname2category.')

    # iteratingly split the largest group until getting n_client
    while n_client > len(category2label):
        largest_group = list(category2label.keys())[0]
        for cat in category2label:
            if len(category2label[cat]) > len(category2label[largest_group]):
                largest_group = cat

        subgroup_labels1, subgroup_labels2 = train_test_split(category2label[largest_group], train_size=0.5, random_state=seed)
        category2label[largest_group + '_1'] = subgroup_labels1
        category2label[largest_group + '_2'] = subgroup_labels2
        del category2label[largest_group]

    # iteratingly merging the smallest group until getting n_client
    while n_client < len(category2label):
        category_count = {cat: len(category2label[cat]) for cat in category2label}
        sorted_categories = sorted(category_count.items(), key=lambda x: x[1])
        smallest_group_1 = sorted_categories[0][0]
        smallest_group_2 = sorted_categories[1][0]

        category2label[smallest_group_1 + ' & ' + smallest_group_2] = category2label[smallest_group_1] + category2label[smallest_group_2]
        del category2label[smallest_group_1]
        del category2label[smallest_group_2]

    client2labels = {i: labels for i, (cat, labels) in enumerate(category2label.items())}
    label2client = {l: c for c, labels in client2labels.items() for l in labels}
    sample2labels = defaultdict(list)
    label2samples = defaultdict(list)
    for i, example_y in enumerate(Y):
        for j, y_val in enumerate(example_y):
            if y_val == 1:
                sample2labels[i].append(j)
                label2samples[j].append(i)

    print(f"{len(client2labels)} categories got.")

    sorted_labels = [item[0] for item in sorted(label2samples.items(), key=lambda item: len(item[1]))]
    sample2client = {}
    client2samples = defaultdict(list)
    for sample_id, sample_labels in sample2labels.items():
        for lb in sorted_labels:
            if lb in sample_labels:
                cat = label2client[lb]
                sample2client[sample_id] = cat
                client2samples[cat].append(sample_id)
                break

    M_client = np.ones_like(M)
    for sample_id, sample_cat in sample2client.items():
        M_client[sample_id, client2labels[sample_cat]] = 0

    M_new = M_client.astype(int) | M.astype(int)

    federated_X = {}
    federated_Y = {}
    federated_M = {}
    original_M = {}
    for cat, sample_ids in client2samples.items():
        federated_X[cat] = [X[i] for i in sample_ids]
        federated_Y[cat] = [Y[i] for i in sample_ids]
        federated_M[cat] = [M_new[i] for i in sample_ids]
        original_M[cat] = [M[i] for i in sample_ids]

    return federated_X, federated_Y, federated_M, original_M, client2labels


def random_k_class(Y, n_client, k):
    # X: a list of data. epoch represents the data of a subject
    client2labels = {}

    candidate_labels_of_client = {c: list(np.unique(Y[c].argmax(1))) for c in range(n_client)}
    for c in range(n_client):
        client2labels[c] = random.sample(candidate_labels_of_client[c], min(len(candidate_labels_of_client[c]), k))

    federated_M = []
    original_M = []
    for c in range(n_client):
        client_M = np.ones_like(Y[c])
        client_M[:, client2labels[c]] = 0.
        for i, (y, m) in enumerate(zip(Y[c], client_M)):
            if np.argmax(y) in client2labels[c]:
                client_M[i, :] = 0.

        federated_M.append(client_M)
        original_M.append(np.zeros_like(Y[c]))

    return federated_M, original_M, client2labels


class SingleLabelDataset(Dataset):
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

    def get_class_weights(self):
        n_classes = len(self.labels[0])
        # Count each class frequency (pos/neg) for each label
        cls_count = np.ones(n_classes)  # avoid nan
        for example_y, example_m in zip(self.labels, self.label_masks):
            cls = example_y.argmax()
            if example_m[cls] == 0.:
                cls_count[example_y.argmax()] += 1

        cls_weight = np.zeros(n_classes)
        for cls in range(n_classes):
            cls_weight[cls] = sum(cls_count) / (cls_count[cls] * len(self.active_targets))
        cls_weight[np.isnan(cls_weight)] = 0

        self.instance_weights = []
        for y in self.labels:
            self.instance_weights.append(sum(cls_weight * y))

    def get_instance_weights(self):
        n_classes = len(self.labels[0])
        # Count each class frequency (pos/neg) for each label
        pos_count = np.ones((n_classes))  # avoid nan
        neg_count = np.ones((n_classes))
        sample_size = np.zeros((n_classes))
        for example_y, example_m in zip(self.labels, self.label_masks):
            for i, (y, m) in enumerate(zip(example_y, example_m)):
                if y == 1:
                    sample_size[i] += 1
                if m == 1:
                    continue
                elif y == 1:
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

        self.instance_weights = np.array(self.instance_weights)

def train_cooccurrence(save_dir, cooccurrence_file, target_names, calibrate=False):
    if os.path.exists(os.path.join(save_dir, 'label_embedding.npy')):
        print('load label embedding from file:', os.path.join(save_dir, 'label_embedding.npy'))
        return np.load(os.path.join(save_dir, 'label_embedding.npy'))
    elif not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cooccurrence_df = pd.read_pickle(open(cooccurrence_file, 'rb'))
    cooccurrence_matrix = cooccurrence_df.loc[target_names][target_names].to_numpy()

    if calibrate:
        threshold = np.nanpercentile(cooccurrence_matrix, 50)
        if threshold < 0:
            print('threshold of weight:', threshold)
            cooccurrence_matrix -= threshold  # np.median(weight_matrix)

    graph = nx.Graph()
    for i in range(len(target_names)):
        graph.add_node(target_names[i])

    for i in tqdm(range(len(target_names)), total=len(target_names)):
        for j in range(len(target_names)):
            if cooccurrence_matrix[i][j] > 0:
                graph.add_edge(target_names[i], target_names[j], weight=cooccurrence_matrix[i][j])

    node2vec = Node2Vec(graph, dimensions=256, walk_length=100, num_walks=200)
    model = node2vec.fit(window=5, min_count=1)

    label_embedding = np.array([model.wv[tn] for tn in target_names])
    np.save(os.path.join(save_dir, 'label_embedding.npy'), label_embedding)

    return label_embedding