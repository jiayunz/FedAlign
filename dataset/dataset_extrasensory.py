import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from torch.nn.utils.rnn import pad_sequence
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .utils import partial_labeling_by_category

def collate_fn(batch):
    batch_data, batch_labels, batch_label_masks, batch_label_masks_true, batch_weight, batch_targets = zip(*batch)

    batch_data = pad_sequence([torch.FloatTensor(x) for x in batch_data], batch_first=True, padding_value=0)
    batch_labels = torch.FloatTensor(batch_labels)
    batch_label_masks = torch.LongTensor(batch_label_masks)
    batch_label_masks_true = torch.LongTensor(batch_label_masks_true)
    batch_weight = torch.FloatTensor(batch_weight)
    batch_targets = torch.LongTensor(batch_targets)

    return batch_data, batch_labels, batch_label_masks, batch_label_masks_true, batch_weight, batch_targets



def load_data(data_path, n_client=1, seed=4321, fold=0):
    data = np.load(os.path.join(data_path, 'data.npz'), allow_pickle=True)
    X = np.array(data['X'])
    Y = np.array(data['Y'])
    feature_names = data['feature_names']
    try:
        target_names = np.array([ln.decode() for ln in data['label_names']])
    except:
        target_names = data['label_names']
    if 'categorical_features' in data:
        categorical_features = data['categorical_features']
    else:
        categorical_features = None
    # count labels
    label_count = np.sum(Y, axis=0)
    imbalance_factor = np.max(label_count) / np.min(label_count)
    print('label_count.shape', label_count.shape, 'imbalance', imbalance_factor)

    M = np.array(data['M'])

    # remove classes with less than 10 samples
    preserve_idx = []
    tn_count = {}
    for i, ln in enumerate(target_names):
        tn_count[ln] = (Y[:, i] == 1).sum()
        if (Y[:, i] == 1).sum() > 10:
            preserve_idx.append(i)
    Y = Y[:, preserve_idx]
    M = M[:, preserve_idx]
    target_names = target_names[preserve_idx]

    targetname2category = {
            'SITTING': 'posture',
            'LYING_DOWN': 'posture',
            'OR_standing': 'posture',
            'FIX_walking': 'posture',
            'BICYCLING': 'posture',
            'FIX_running': 'posture',

            'PHONE_ON_TABLE': 'phone',
            'PHONE_IN_POCKET': 'phone',
            'PHONE_IN_HAND': 'phone',
            'PHONE_IN_BAG': 'phone',

            'WITH_CO-WORKERS': 'accompany',
            'WITH_FRIENDS': 'accompany',

            'OR_outside': 'environment',
            'OR_indoors': 'environment',
            'AT_SCHOOL': 'environment',
            'AT_A_PARTY': 'environment',
            'AT_THE_GYM': 'environment',
            'AT_A_BAR': 'environment',
            'LOC_beach': 'environment',
            'LOC_home': 'environment',
            'LOC_main_workplace': 'environment',
            'FIX_restaurant': 'environment',
            'IN_CLASS': 'environment',
            'IN_A_MEETING': 'environment',
            'ON_A_BUS': 'environment',
            'IN_A_CAR': 'environment',
            'ELEVATOR': 'environment',
            'TOILET': 'environment',

            'COOKING': 'activity',
            'CLEANING': 'activity',
            'DOING_LAUNDRY': 'activity',
            'WASHING_DISHES': 'activity',
            'GROOMING': 'activity',
            'DRESSING': 'activity',
            'SLEEPING': 'activity',
            'EATING': 'activity',
            'BATHING_-_SHOWER': 'activity',
            'LAB_WORK': 'activity',
            'COMPUTER_WORK': 'activity',
            'SURFING_THE_INTERNET': 'activity',
            'OR_exercise': 'activity',
            'DRIVE_-_I_M_THE_DRIVER': 'activity',
            'DRIVE_-_I_M_A_PASSENGER': 'activity',
            'SHOPPING': 'activity',
            'TALKING': 'activity',
            'WATCHING_TV': 'activity',
            'DRINKING__ALCOHOL_': 'activity',
            'SINGING': 'activity',
            'STROLLING': 'activity',
            'STAIRS_-_GOING_UP': 'activity',
            'STAIRS_-_GOING_DOWN': 'activity'
        }
    print('loaded X and Y')

    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    for i, (train_indices, test_indices) in enumerate(kf.split(range(len(X)))):
        if i == fold:
            X_train, X_test, Y_train, Y_test, M_train, M_test = X[train_indices], X[test_indices], Y[train_indices], Y[test_indices], M[train_indices], M[test_indices]
            break
    else:
        raise ValueError(f'fold {fold} is larger than 5. Set a smaller fold.')

    if n_client > 1:
        X_train, Y_train, M_train, M_train_orig, client2labels = partial_labeling_by_category(X_train, Y_train, M_train, n_client, targetname2category, target_names, seed=seed)
    else:
        X_train, Y_train, M_train, M_train_orig = [X_train], [Y_train], [M_train], [M_train]
        client2labels = {0: range(len(M_train[0][0]))}

    print(client2labels)
    trainData = []

    X_val = []
    Y_val = []
    M_val = []
    M_val_orig = []

    testData = MyDataset(X_test, Y_test, M_test, M_true=M_test, max_len=10, active_targets=range(len(M_test[0])), feature_names=feature_names, target_names=target_names, categorical_features=categorical_features)

    for c in range(n_client):
        X_train_c, X_val_c, Y_train_c, Y_val_c, M_train_c, M_val_c, M_train_orig_c, M_val_orig_c = train_test_split(X_train[c], Y_train[c], M_train[c], M_train_orig[c], test_size=0.25, random_state=seed)
        X_val.extend(X_val_c)
        Y_val.extend(Y_val_c)
        M_val.extend(M_val_c)
        M_val_orig.extend(M_val_orig_c)

        trainData.append(MyDataset(X_train_c, Y_train_c, M_train_c, M_true=M_train_orig_c, max_len=10, active_targets=[l for l in client2labels[c]], feature_names=feature_names, target_names=target_names, categorical_features=categorical_features, preprocessor=testData.preprocessor))

    valData = MyDataset(X_val, Y_val, M_val, M_true=M_val_orig, max_len=10, active_targets=range(len(M_test[0])), feature_names=feature_names, target_names=target_names, categorical_features=categorical_features, preprocessor=testData.preprocessor)

    return trainData, valData, testData


class MyDataset(Dataset):
    def __init__(self, X, Y, M, M_true, max_len, active_targets=None, feature_names=None, target_names=None, categorical_features=None, preprocessor=None):
        self.data = []
        self.labels = []
        self.label_masks = []
        self.label_masks_true = []

        for x, y, m, m_true in zip(X, Y, M, M_true):
            if len(x) > max_len:
                start = 0
                while start < len(x):
                    self.data.append(x[start:start + max_len])
                    self.labels.append(y)
                    self.label_masks.append(m)
                    self.label_masks_true.append(m_true)
                    start += max_len
            else:
                self.data.append(x)
                self.labels.append(y)
                self.label_masks.append(m)
                self.label_masks_true.append(m_true)

        if active_targets is None:
            self.active_targets = np.where(np.any(self.label_masks == 0, dim=0))[0]
        else:
            self.active_targets = active_targets

        self.get_instance_weights()
        self.data_len = np.array([len(x) for x in self.data])
        self.feature_names = feature_names
        self.target_names = target_names
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.preprocessor = preprocessor
        self.preprocessing()

        self.data = self.data
        self.labels = np.array(self.labels)
        self.label_masks = np.array(self.label_masks)
        self.label_masks_true = np.array(self.label_masks_true)
        self.instance_weights = np.array(self.instance_weights)


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


    def preprocessing(self):
        # (n_samples, seq_len, features)
        concat_data = []
        for x in self.data:
            concat_data.extend(x)

        concat_data = pd.DataFrame(concat_data, columns=self.feature_names)

        selected_indices = np.random.choice(range(len(concat_data)), max(int(0.1 * len(concat_data)), min(10000, len(concat_data))), replace=False)

        if self.preprocessor is None:
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            numeric_indices = [i for i, ln in enumerate(self.feature_names) if ln not in self.categorical_features]
            categorical_indices = [i for i, ln in enumerate(self.feature_names) if ln in self.categorical_features]

            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
                ("scaler", StandardScaler())
            ])

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_indices),
                    ("cat", categorical_transformer, categorical_indices),
                ]
            )
            self.preprocessor.fit(concat_data.loc[selected_indices])

        try:
            concat_data = self.preprocessor.transform(concat_data).todense()
        except:
            concat_data = self.preprocessor.transform(concat_data)

        processed_data = []
        start = 0
        for l in self.data_len:
            processed_data.append(concat_data[start:start + l])
            start += l

        self.data = processed_data