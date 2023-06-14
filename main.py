import os
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import random
from collections import defaultdict
from copy import deepcopy
import torch
from torch.utils.data import DataLoader

from dataset.utils import train_cooccurrence
from build_model import build_model
from evaluation import display_results

DATA_DIR = 'data/'
MODEL_DIR = 'model/'

def run(args, fold, seed):
    save_dir = os.path.join(MODEL_DIR, f"{args.task}/seed{seed}/")
    cooccurrence_dir = os.path.join(MODEL_DIR, f"{args.task}/seed{seed}")

    if args.dataset == 'extrasensory':
        data_path = os.path.join(DATA_DIR, 'ExtraSensory')
        cooccurrence_path = os.path.join(DATA_DIR, 'cooccurrence/cooccurrence_extrasensory.pkl')
        trainData, valData, testData = load_data(data_path, args.n_client, seed=seed, fold=fold)
        target_names = testData.target_names
        data_feature_size = np.shape(testData.data[0])[-1]
        save_dir = os.path.join(save_dir, f"fold{fold}")

    elif args.dataset == 'mimic3':
        data_path = os.path.join(DATA_DIR, 'MIMIC/medical-codes')
        cooccurrence_path = os.path.join(DATA_DIR, 'cooccurrence/cooccurrence_mimic3.pkl')
        trainData, valData, testData = load_data(data_path, args.n_client, seed=seed)
        target_names = testData.target_names
        data_feature_size = len(testData.vocab)

    elif args.dataset == 'pamap2':
        data_path = os.path.join(DATA_DIR, 'PAMAP2/pamap2_data_100.pkl')
        label_path = os.path.join(DATA_DIR, 'PAMAP2/pamap2_label_100.pkl')
        cooccurrence_path = os.path.join(DATA_DIR, 'cooccurrence/cooccurrence_pamap2.pkl')
        trainData, valData, testData = load_data(data_path, label_path, k_class=args.k_class, seed=seed)
        target_names = testData.target_names
        data_feature_size = np.shape(testData.data[0])[-1]

    elif args.dataset == 'r8':
        data_path = os.path.join(DATA_DIR, 'Reuters-21578')
        cooccurrence_path = os.path.join(DATA_DIR, 'cooccurrence/cooccurrence_reuters.pkl')
        trainData, valData, testData = load_data(data_path, args.n_client, k_class=args.k_class, seed=seed)
        target_names = testData.target_names
        data_feature_size = len(testData.vocab)

    else:
        raise ValueError('Wrong dataset.')

    if args.no_pretrain:
        pretrained_embedding = None
    else:
        pretrained_embedding = train_cooccurrence(cooccurrence_dir, cooccurrence_path, target_names, calibrate=False)
        pretrained_embedding = torch.FloatTensor(pretrained_embedding).to(args.device)
        print('pretrained embedding matrix:', pretrained_embedding.shape)

    print(pretrained_embedding)
    print('# of samples in trainData:', [len(td) for td in trainData])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('save_dir:', save_dir)

    encoded_labels = torch.arange(0, end=len(testData.target_names)).to(args.device)

    print(f'n_class: {len(target_names)}, n_vocab: {encoded_labels.max() + 1}, n_feature: {data_feature_size}')
    global_model = build_model(
        use_label_encoder=args.fedalign,
        hidden_dim=256,
        data_feature_size=data_feature_size,
        n_class=len(target_names),
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        dropout=0.5,
        pretrained_embedding=pretrained_embedding,
        do_input_embedding=args.do_input_embedding
    )
    client_models = [deepcopy(global_model) for _ in range(args.n_client)]

    global_model = global_model.to(args.device)
    client_models = [model.to(args.device) for model in client_models]

    framework = Framework(args, global_model, client_models, encoded_labels, target_names, metrics=metrics)
    train_log = {'args': args, 'test_result': []}
    framework.train(args, trainData, valData, testData, collate_fn, train_log, save_dir)

    best_model = torch.load(os.path.join(save_dir, f'model.pt'))

    test_loader = DataLoader(testData, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1)
    test_true, test_pred, test_mask = framework.evaluate(best_model, test_loader)
    results = calculate_metrics(test_true, test_pred, test_mask)

    del trainData
    del valData
    del testData
    del global_model
    del client_models
    del best_model
    del framework

    return results


def parse_args():
    # default setting is for extrasensory
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default="5", help="gpu id")
    parser.add_argument('--random_seeds', type=int, default=[4321, 4322, 4323, 4324, 4325], help="random seed")
    # task
    parser.add_argument('-t', '--task', choices=['es-5', 'es-15', 'es-25', 'mimic3', 'pamap2', 'r8'], default='mimic3', help="task name")
    parser.add_argument('-c', '--n_client', type=int, default=10, help="number of clients")
    parser.add_argument('--k_class', type=int, default=10, help="number of random class per client")
    # FL setting
    parser.add_argument('--sample_clients', type=int, default=5, help="number of clients join training at each round")
    parser.add_argument('-e', '--epochs', type=int, default=5, help="number of training epochs per round")
    parser.add_argument('-r', '--rounds', type=int, default=50, help="number of iteration rounds")
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument('--fedalign', action='store_true')
    parser.add_argument('--data_lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--label_lr', type=float, default=0.001, help="learning rate")
    # pseudo labeling
    parser.add_argument('--pos', type=float, default=99.9, help="percentile of similarity of positive pseudo samples")
    parser.add_argument('--neg', type=float, default=99.9, help="percentile of similarity of negative pseudo samples")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = ['F1', 'ACC']

    if args.task.startswith('es-'):
        args.dataset = 'extrasensory'
        args.do_input_embedding = False
        args.normalize = False
        if args.task == 'es-5':
            args.n_client = 5
        elif args.task == 'es-15':
            args.n_client = 15
        elif args.task == 'es-25':
            args.n_client = 25

        from framework import MLCFramework as Framework
        from evaluation import calculate_MLC_metrics as calculate_metrics
        from dataset.dataset_extrasensory import load_data, collate_fn

    elif args.task == 'mimic3':
        args.dataset = 'mimic3'
        args.n_client = 10
        args.rounds = 100
        args.do_input_embedding = True
        args.normalize = False
        args.label_lr = 0.005

        from framework import MLCFramework as Framework
        from evaluation import calculate_MLC_metrics as calculate_metrics
        from dataset.dataset_mimic3 import load_data, collate_fn

    elif args.task == 'pamap2':
        args.dataset = 'pamap2'
        args.k_class = 5
        args.do_input_embedding = False
        args.normalize = True
        args.n_client = 9
        args.label_lr = 0.005
        args.pos = 99
        args.neg = 50

        from framework import SLCFramework as Framework
        from evaluation import calculate_SLC_metrics as calculate_metrics
        from dataset.dataset_pamap2 import load_data, collate_fn

    elif args.task == 'r8':
        args.dataset = 'r8'
        args.k_class = 3
        args.do_input_embedding = True
        args.n_client = 8
        args.normalize = True
        args.pos = 99
        args.neg = 50

        from framework import SLCFramework as Framework
        from evaluation import calculate_SLC_metrics as calculate_metrics
        from dataset.dataset_r8 import load_data, collate_fn

    else:
        raise NotImplementedError('Wong dataset')

    results = defaultdict(list)

    for fold, seed in zip(range(5), args.random_seeds):
        set_seed(seed)
        print(args)
        print(f'#### Run Experiments on seed {seed} ####')
        seed_results = run(args, fold, seed)
        for m in metrics:
            results[m].append(seed_results[m])

        display_results({m: np.average(results[m]) for m in metrics}, metrics)
