import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score

MIN_FLOAT = 1e-12

def calculate_MLC_metrics(y_true, y_pred, y_mask):
    support = defaultdict(int)
    for sample_y, sample_m in zip(y_true, y_mask):
        for cls, (y_val, m_val) in enumerate(zip(sample_y, sample_m)):
            support[cls] += int(m_val == 0)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true = y_true == 1
    pred = y_pred > 0.5
    mask = np.array(y_mask)

    class_results = dict({
        'F1': [],
        'ACC': [],
        'support': []
    })

    for i in range(len(y_true[0])):
        class_results['support'].append(support[i])
        valid_indices = np.where(mask[:, i] != 1)[0]
        if support[i] > 0:
            i_true = true[:, i][valid_indices]
            i_pred = pred[:, i][valid_indices]
            class_results['F1'].append(f1_score(y_true=i_true, y_pred=i_pred))
            class_results['ACC'].append(accuracy_score(y_true=i_true, y_pred=i_pred))
        else:
            for metric in class_results:
                if metric != 'support':
                    class_results[metric].append(np.nan)

    class_results['support'] = np.array(class_results['support'])

    all_results = {
        'ACC': np.nanmean(class_results['ACC']),
        'F1': np.nanmean(class_results['F1']),
        'support': sum(support.values())
    }
    return all_results



def calculate_SLC_metrics(y_true, y_pred, y_mask):
    support = defaultdict(int)
    for sample_y, sample_m in zip(y_true, y_mask):
        for cls, (y_val, m_val) in enumerate(zip(sample_y, sample_m)):
            support[cls] += int(m_val == 0)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true_bool = y_true.argmax(-1)
    else:
        y_true_bool = y_true

    if len(y_pred.shape) > 1:
        y_pred_bool = y_pred.argmax(-1)
    else:
        y_pred_bool = y_pred

    all_results = {
        'F1': f1_score(y_true_bool, y_pred_bool, average='macro'),
        'ACC': accuracy_score(y_true_bool, y_pred_bool)
    }

    return all_results

def display_results(results, metrics=['BA', 'F1', 'ACC']):
    print('{0:>20}'.format("") + ' '.join(['%10s']*len(metrics)) % tuple([m for m in metrics]))
    print('{0:>20}'.format("AVG") + ' '.join(['%10.4f'] * len(metrics)) % tuple([results[m] for m in metrics]))

    return [results[m] for m in metrics]