import os
import torch
from torch import nn
import numpy as np
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from collections import defaultdict

from evaluation import calculate_MLC_metrics, calculate_SLC_metrics, display_results


class SoftCrossEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super(SoftCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, target, mask):
        log_likelihood = F.log_softmax(inputs, dim=1)
        batch_loss = - torch.sum((1 - mask) * target * log_likelihood, dim=1)

        if self.reduction == 'mean':
            return torch.mean(batch_loss)
        elif self.reduction == 'sum':
            return torch.sum(batch_loss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')

class BasicFramework():
    def __init__(self, args, global_model, client_models, encoded_labels, target_names, metrics=['F1', 'ACC']):
        self.global_model = global_model
        print(self.global_model)
        self.client_models = [model for model in client_models]
        self.encoded_labels = encoded_labels
        self.target_names = target_names
        self.device = args.device
        self.n_clients = len(self.client_models)
        self.metrics = metrics
        self.fedalign = args.fedalign

    def server_aggregate(self, client_class_weights):
        update_weights = torch.where(client_class_weights.sum(-1) > 0, 1, 0)
        update_clients = torch.where(update_weights > 0)[0]
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            if k == 'label_encoder.weight':
                for i, _ in enumerate(self.target_names):
                    if len([c for c in update_clients if i in self.on_device_targets[c]]):
                        k_update_weights = torch.FloatTensor([[update_weights[c]] for c in update_clients if i in self.on_device_targets[c]])
                        k_update_weights = k_update_weights.to(self.device) / k_update_weights.sum()
                        global_dict[k][i] = (k_update_weights * torch.stack([self.client_models[c].state_dict()[k][i].float() for c in update_clients if i in self.on_device_targets[c]], 0)).sum(0)
            else:
                k_update_weights = deepcopy(update_weights)
                for _ in global_dict[k].shape:
                    k_update_weights = k_update_weights.unsqueeze(-1)
                k_update_weights = k_update_weights.to(self.device) / update_weights.sum()
                global_dict[k] = (k_update_weights * torch.stack([self.client_models[c].state_dict()[k].float() for c in range(self.n_clients)], 0)).sum(0)

        self.global_model.load_state_dict(global_dict)

    def client_update(self):
        for model in self.client_models:
            model.load_state_dict(deepcopy(self.global_model.state_dict()))

    def train(self, args, trainData, valData, testData, collate_fn, train_log, save_dir):
        self.on_device_targets = [td.active_targets for td in trainData]

        client_optimizers = []
        for client_model in self.client_models:
            if args.fedalign:
                for param in client_model.label_encoder.parameters():
                    param.requires_grad = False
                optimizer = {'data': torch.optim.Adam(filter(lambda p: p.requires_grad, client_model.parameters()), lr=args.data_lr)}
                for param in client_model.parameters():
                    param.requires_grad = True
                for param in client_model.data_encoder.parameters():
                    param.requires_grad = False
                optimizer['label'] = torch.optim.Adam(filter(lambda p: p.requires_grad, client_model.parameters()), lr=args.label_lr)
                client_optimizers.append(optimizer)
            else:
                client_optimizers.append(torch.optim.Adam(client_model.parameters(), lr=args.label_lr))

        train_log['metrics'] = self.metrics

        val_loader = DataLoader(valData, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1)
        test_loader = DataLoader(testData, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1)
        test_true, test_pred, test_mask = self.evaluate(self.global_model, test_loader)
        test_results = self.calculate_metrics(test_true, test_pred, test_mask)
        print(f'init score: round {-1}')
        display_results(test_results, metrics=self.metrics)

        for r in range(args.rounds):
            selected_clients = np.random.permutation(range(self.n_clients))[:args.sample_clients]
            # number of samples in each class
            client_class_weights = torch.zeros((self.n_clients, len(self.target_names)))
            for i in selected_clients:
                clientData = trainData[i]
                if self.fedalign and self.n_clients > 1:
                    val_scores = self.validate(self.global_model, val_loader)
                    clientData = self.anchor_guided_alignment(
                        model=self.client_models[i],
                        train_loader=DataLoader(clientData, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1),
                        orig_active_targets=clientData.active_targets,
                        pos_percentile=args.pos,
                        neg_percentile=args.neg,
                        val_scores=val_scores
                    )

                client_class_weights[i] = torch.Tensor(clientData.num_samples)
                client_loader = DataLoader(clientData, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
                self.train_client(self.client_models[i], client_optimizers[i], client_loader, args.epochs)

            self.server_aggregate(client_class_weights)
            self.client_update()

            test_true, test_pred, test_mask = self.evaluate(self.global_model, test_loader)
            test_results = self.calculate_metrics(test_true, test_pred, test_mask)
            print(f'[TRAIN] Round {r}, selected clients:', selected_clients)
            test_scores = display_results(test_results, metrics=self.metrics)
            train_log['test_result'].append(test_scores)

            torch.save(self.global_model, os.path.join(save_dir, f'model.pt'))
            pickle.dump(train_log, open(os.path.join(save_dir, f'train_log.pkl'), 'wb'))

    def train_client(self, model, optimizer, client_loader, epochs):
        model.train()
        for e in range(epochs):
            for sample, label, label_mask, _, weight, client_targets in client_loader:
                if self.fedalign:
                    # train data encoder
                    for param in model.parameters():
                        param.requires_grad = True
                    for param in model.label_encoder.parameters():
                        param.requires_grad = False
                    self.train_one_batch(model, sample, label, label_mask, weight, optimizer['data'])
                    # train label encoder
                    for param in model.parameters():
                        param.requires_grad = True
                    for param in model.data_encoder.parameters():
                        param.requires_grad = False
                    self.train_one_batch(model, sample, label, label_mask, weight, optimizer['label'])
                else:
                    self.train_one_batch(model, sample, label, label_mask, weight, optimizer)

    def validate(self, model, data_loader):
        scores = defaultdict(list)
        # valData is the concatenation of all validation datasets with appropriate masks
        y_true, y_pred, y_mask = self.evaluate(model, data_loader)
        for i in range(y_mask.shape[-1]):
            results = self.calculate_metrics(y_true[:, i:i+1], y_pred[:, i:i+1], y_mask[:, i:i+1])
            scores[i].append([results[m] for m in self.metrics])
        class_scores = []
        for i, tn in enumerate(self.target_names):
            class_scores.append([np.nanmean(np.array(scores[i])[:, m]) for m in range(len(self.metrics))])

        return class_scores

    def evaluate(self, model, data_loader):
        y_pred = []
        y_true = []
        y_mask = []

        model.eval()
        with torch.no_grad():
            for sample, label, label_mask, _, weight, client_targets in data_loader:
                sample = sample.to(self.device).transpose(0, 1)
                label = label.to(self.device, dtype=torch.float)
                label_mask = label_mask.to(self.device, dtype=torch.long)

                out = model(sample, self.encoded_labels, normalize_label=self.normalize_label)
                out = self.activation(out) # softmax or sigmoid

                y_pred.extend(out.cpu().numpy())
                y_true.extend(label.cpu().numpy())
                y_mask.extend(label_mask.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_mask = np.array(y_mask)

        return y_true, y_pred, y_mask

    def calculate_metrics(self, val_true, val_pred, val_mask):
        return defaultdict(float)

class MLCFramework(BasicFramework): # multilabel classification
    def __init__(self, args, global_model, client_models, encoded_labels, target_names, metrics):
        super(MLCFramework, self).__init__(args, global_model, client_models, encoded_labels, target_names, metrics)
        self.normalize_label = False

    def anchor_guided_alignment(self, model, train_loader, orig_active_targets, pos_percentile, neg_percentile, val_scores):
        # model: client model / after aggregation, same as global model
        # val_scores: validate score of each class
        missing_targets = [t for t in range(len(self.target_names)) if t not in orig_active_targets]
        X_pseudo = []
        Y_pseudo = []
        M_pseudo = []
        M_true = []  # only for evaluation
        Y_sim = []

        model.eval()
        with torch.no_grad():
            class_anchors = model.label_encoder(self.encoded_labels)  # [n_class, emb_dim]
            for data, label, label_mask, label_mask_true, weight, client_targets in train_loader:
                sample = data.to(self.device).transpose(0, 1)
                data_rep = model.data_encoder(sample) # [batch_size, emb_dim]
                similarity = torch.mul(data_rep.unsqueeze(1), class_anchors.repeat(data_rep.size(0), 1, 1)).sum(-1, keepdims=True).reshape((-1, len(self.target_names)))

                X_pseudo.extend(data.cpu().numpy())
                Y_sim.extend(similarity.cpu().numpy())
                Y_pseudo.extend(label.cpu().numpy())
                M_pseudo.extend(label_mask.cpu().numpy())
                M_true.extend(label_mask_true.cpu().numpy())

        X_pseudo = X_pseudo
        Y_pseudo = np.array(Y_pseudo) # for missing classes, use prediction, for active targets, use ground truth
        Y_sim = np.array(Y_sim)
        M_pseudo = np.array(M_pseudo)
        M_true = np.array(M_true)

        new_active_targets = deepcopy(orig_active_targets)
        for i in range(Y_pseudo.shape[-1]):
            if i in missing_targets:
                confidence = Y_sim[:, i]
                pos_threshold = max(0.5, np.percentile(confidence, pos_percentile))
                neg_threshold = min(0.5, np.percentile(confidence, neg_percentile))
                metric2idx = {m: m_idx for m_idx, m in enumerate(self.metrics)}
                pos_pseudo_index = []
                if val_scores[i][metric2idx['F1']] > 0.75:
                    pos_pseudo_index = list(np.where(Y_sim[:, i] > pos_threshold)[0])
                neg_pseudo_index = list(np.where(Y_sim[:, i] < neg_threshold)[0])

                M_pseudo[pos_pseudo_index, i] = 0
                Y_pseudo[pos_pseudo_index, i] = 1.
                M_pseudo[neg_pseudo_index, i] = 0
                Y_pseudo[neg_pseudo_index, i] = 0.

                if len(neg_pseudo_index) + len(pos_pseudo_index):
                    new_active_targets.append(i)

        clientData = MixupDataset(X_pseudo, Y_pseudo, M_pseudo, M_true, target_names=self.target_names, active_targets=new_active_targets)

        return clientData

    def train_one_batch(self, model, sample, label, mask, weight, optimizer):
        sample = sample.to(self.device).transpose(0, 1)
        label = label.to(self.device, dtype=torch.float)
        weight = weight.to(self.device, dtype=torch.float)
        optimizer.zero_grad()
        out = model(sample, self.encoded_labels, normalize_label=self.normalize_label)
        # weight is used to control the mask
        loss = F.binary_cross_entropy_with_logits(out, label, weight=weight)
        loss.backward()
        optimizer.step()
        return loss.item()

    def activation(self, output):
        return torch.sigmoid(output)

    def calculate_metrics(self, val_true, val_pred, val_mask):
        return calculate_MLC_metrics(val_true, val_pred, val_mask)

class SLCFramework(BasicFramework): # single label classification
    def __init__(self, args, global_model, client_models, encoded_labels, target_names, metrics):
        super(SLCFramework, self).__init__(args, global_model, client_models, encoded_labels, target_names, metrics)
        self.normalize_label = True

    def anchor_guided_alignment(self, model, train_loader, orig_active_targets, pos_percentile, neg_percentile, val_scores):
        # model: client model / after aggregation, same as global model
        # val_scores: validate score of each class

        missing_targets = [t for t in range(len(self.target_names)) if t not in orig_active_targets]
        X_pseudo = []
        Y_pseudo = []
        M_pseudo = []
        M_true = []  # only for evaluation
        Y_sim = []

        model.eval()
        with torch.no_grad():
            class_anchors = model.label_encoder(self.encoded_labels)
            z_class_norm = class_anchors / class_anchors.norm(p=2, dim=-1, keepdim=True)
            for data, label, label_mask, label_mask_true, weight, client_targets in train_loader:
                sample = data.to(self.device).transpose(0, 1)
                data_rep = model.data_encoder(sample)
                z_data_norm = data_rep / data_rep.norm(p=2, dim=-1, keepdim=True)
                similarity = torch.mul(z_data_norm.unsqueeze(1), z_class_norm.repeat(data_rep.size(0), 1, 1)).sum(-1, keepdims=True).reshape((-1, len(self.target_names)))

                X_pseudo.extend(data.cpu().numpy())
                Y_sim.extend(similarity.cpu().numpy())
                Y_pseudo.extend(label.cpu().numpy())
                M_pseudo.extend(label_mask.cpu().numpy())
                M_true.extend(label_mask_true.cpu().numpy())

        X_pseudo = X_pseudo
        Y_sim = np.array(Y_sim)
        M_pseudo = np.array(M_pseudo)
        M_true = np.array(M_true)
        Y_pseudo = np.array(Y_pseudo)

        new_active_targets = deepcopy(orig_active_targets)
        masked_sample_index = set(np.where((Y_pseudo * (1 - M_pseudo)).sum(-1) == 0)[0])
        for i in range(Y_pseudo.shape[-1]):
            if i in missing_targets:
                pos_threshold = np.percentile(Y_sim[:, i], pos_percentile)
                neg_threshold = np.percentile(Y_sim[:, i], neg_percentile)
                pos_pseudo_index = list(set(np.where(Y_sim.argmax(-1) == i)[0]) & set(np.where(Y_sim[:, i] > pos_threshold)[0]) & masked_sample_index)
                neg_pseudo_index = list(set(np.where(Y_sim.argmax(-1) != i)[0]) & set(np.where(Y_sim[:, i] < neg_threshold)[0]) & masked_sample_index)
                M_pseudo[pos_pseudo_index, i] = 0
                Y_pseudo[pos_pseudo_index] = 0.
                Y_pseudo[pos_pseudo_index, i] = 1.
                M_pseudo[neg_pseudo_index, i] = 0
                Y_pseudo[neg_pseudo_index, i] = 0.

                if len(pos_pseudo_index) + len(neg_pseudo_index):
                    new_active_targets.append(i)

        clientData = MixupDataset(X_pseudo, Y_pseudo, M_pseudo, M_true, target_names=self.target_names, active_targets=new_active_targets)
        return clientData

    def train_one_batch(self, model, sample, label, mask, weight, optimizer):
        criterion = SoftCrossEntropy()
        sample = sample.to(self.device).transpose(0, 1)
        label = label.to(self.device, dtype=torch.float)
        mask = mask.to(self.device, dtype=torch.float)
        optimizer.zero_grad()
        out = model(sample, self.encoded_labels, normalize_label=self.normalize_label)
        loss = criterion(out, label, mask)
        loss.backward()
        optimizer.step()
        return loss.item()

    def activation(self, output):
        return torch.softmax(output, dim=-1)

    def calculate_metrics(self, val_true, val_pred, val_mask):
        return calculate_SLC_metrics(val_true, val_pred, val_mask)

class MixupDataset(Dataset):
    def __init__(self, data, labels, masks, masks_true, target_names, active_targets=None):
        self.data = data
        self.labels = labels
        self.label_masks = masks
        self.label_masks_true = masks_true
        self.target_names = target_names
        self.get_instance_weights()
        if active_targets is not None:
            self.active_targets = active_targets
        else:
            self.active_targets = np.array(range(labels.shape[1]))

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
        # number of samples in each class
        self.num_samples = pos_count - 1
        self.pos_weight = neg_count / (pos_count + neg_count)
        self.neg_weight = pos_count / (pos_count + neg_count)

        self.instance_weights = []
        for y, m in zip(self.labels, self.label_masks):
            weight = (y * self.pos_weight + (1 - y) * self.neg_weight) * (1 - m)
            self.instance_weights.append(weight)

        self.instance_weights = np.array(self.instance_weights)

        self.priorlist = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        label_masks = self.label_masks[idx]
        label_masks_true = self.label_masks_true[idx]
        instance_weights = self.instance_weights[idx]
        return data, labels, label_masks, label_masks_true, instance_weights, self.active_targets
