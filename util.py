import numpy as np
from collections import Counter, defaultdict
import torch
import pickle

np.random.seed(1234)

def precision_at_k(r, k):
    k = min(len(r), k)
    r = np.asarray(r)[:k] != 0
    return np.mean(r)

def average_precision(r, k):
    k = min(k, len(r))
    r = np.asarray(r) != 0
    out = [precision_at_k(r, i + 1) for i in range(k) if r[i]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(rs, k):
    return np.mean([average_precision(r, k) for r in rs])

def mean_precision_at_k(rs, k):
    return np.mean([precision_at_k(r, k) for r in rs])

def recommend(edge_index, y, pred):
    users = set(edge_index[1])
    recommended_list = defaultdict(list)
    for item_i, user_i, y_i, pred_i in zip(edge_index[0], edge_index[1], y, pred):
        recommended_list[user_i].append((pred_i, y_i))

    output_lists = []
    for user_i, list_i in recommended_list.items():
        recommend = sorted(list_i, key=lambda x: x[0], reverse=True)
        output_lists.append([x[1] for x in recommend])

    precision_at_5, map_at_5, precision_at_10, map_at_10, precision_at_20, map_at_20 = mean_precision_at_k(
        output_lists, 5), mean_average_precision(
        output_lists, 5), mean_precision_at_k(
        output_lists, 10),mean_average_precision(
        output_lists, 10), mean_precision_at_k(
        output_lists, 20), mean_average_precision(
        output_lists, 20)

    return precision_at_5, map_at_5, precision_at_10, map_at_10, precision_at_20, map_at_20


def get_largest_interacted_number(data):
    max_num = 0
    for user, hist in data.items():
        max_num = max(len(hist), max_num)
    return max_num

def zero_padding(data, padto, feat, feat_dim):
    X, y = [], []
    num_user = len(data)
    item_w_feat = set(feat.docvecs.doctags.keys())



    for user, hist in data.items():
        item_list = []
        label_list = []
        hist = sorted(hist, key=lambda x: x[2], reverse=True)


        for v, r, p, emb in hist:
            if v in item_w_feat:
                item_list.append(list(emb))
                label_list.append(r)

        X.append(item_list)
        y.append(label_list)


    X_pad, y_pad, mask = [], [], []
    for x_i, y_i in zip(X, y):
        if len(x_i) < padto:
            mask.append(len(x_i))

            x_i = x_i + [np.zeros(feat_dim)] * (padto - len(x_i))
            y_i = y_i + [0] * (padto - len(y_i))
        else:
            mask.append(padto)

            x_i = x_i[:padto]
            y_i = y_i[:padto]


        X_pad.append(x_i)
        y_pad.append(y_i)

    
    return torch.FloatTensor(X_pad), torch.FloatTensor(y_pad), torch.LongTensor(mask)

def batch_iter(x, y, mask, batch_size=16, shuffle=False):
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1
        if shuffle:
            indices = np.random.permutation(np.arange(data_len))
        else:
            indices = np.arange(data_len)
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x[indices[start_id:end_id]], y[indices[start_id:end_id]], mask[indices[start_id:end_id]]




if __name__=='__main__':
    print(average_precision([1, 1, 0, 1, 0, 1, 0, 0, 0, 1], 10))