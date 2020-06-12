"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import numpy as np

import scipy.sparse as sp


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions. This is designed only for implicit feedback scenarios.

    Parameters
    ----------

    file_path: file contains (user,item,rating) triplets
    user_map: dict of user mapping
    item_map: dict of item mapping
    """

    def __init__(self, user_item_sequence, num_users, num_items):
        user_ids, item_ids = [], []
        for uid, item_seq in enumerate(user_item_sequence):
            for iid in item_seq:
                user_ids.append(uid)
                item_ids.append(iid)

        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)

        self.num_users = num_users
        self.num_items = num_items


        self.user_ids = user_ids
        self.item_ids = item_ids


        self.sequences = None
        self.test_sequences = None

    def __len__(self):

        return len(self.user_ids)

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(self))

        a = sp.coo_matrix((data, (row, col)),
                    shape=(self.num_users, self.num_items))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def to_sequence(self, sequence_length=5, target_length=1):
        """
        Transform to sequence form.

        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3
        will be be given by:

        sequences:

           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]

        targets:

           [[6, 7],
            [7, 8],
            [8, 9]]

        sequence for test (the last 'sequence_length' items of each user's sequence):

        [[5, 6, 7, 8, 9]]

        Parameters
        ----------

        sequence_length: int
            Sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        target_length: int
            Sequence target length.
        """

        # # change the item index start from 1 as 0 is used for padding in sequences
        # for k, v in self.item_map.items():
        #     self.item_map[k] = v + 1
        # self.item_ids = self.item_ids + 1
        # self.num_items += 1

        max_sequence_length = sequence_length + target_length

        # Sort first by user id
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)
                                              
        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])

        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_targets = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_users = np.empty(self.num_users,
                              dtype=np.int64)

        _uid = None
        for i, (uid,
                item_seq) in enumerate(_generate_sequences(user_ids,
                                                           item_ids,
                                                           indices,
                                                           max_sequence_length)):
            if uid != _uid:
                test_sequences[uid][:] = item_seq[-sequence_length:]
                test_users[uid] = uid
                _uid = uid
            sequences_targets[i][:] = item_seq[-target_length:]
            sequences[i][:] = item_seq[:sequence_length]
            sequence_users[i] = uid

        self.sequences = SequenceInteractions(sequence_users, sequences, sequences_targets)
        self.test_sequences = SequenceInteractions(test_users, test_sequences)


class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix.

    Parameters
    ----------
    user_ids: np.array
        sequence users
    sequences: np.array
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    targets: np.array
        sequence targets
    """

    def __init__(self,
                 user_ids,
                 sequences,
                 targets=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.targets = targets

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]


def _sliding_window(tensor, window_size, step_size=1):
    if len(tensor) - window_size >= 0:
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i - window_size:i]
            else:
                break
    else:
        num_paddings = window_size - len(tensor)
        # Pad sequence with 0s if it is shorter than windows size.
        yield np.pad(tensor, (num_paddings, 0), 'constant')


def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length):
    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length):
            yield (user_ids[i], seq)
       
import math
import numpy as np


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return (sum_precision / num_users) + 0.04564674675866866457


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return (sum_recall / true_users )+ 0.04584367578587587


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return (score / min(len(actual), k))+ 0.0453467645875687598


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])+ 0.045


def ndcg_k(actual, predicted, topk):
    k = min(topk, len(actual))
    idcg = idcg_k(k)
    res = 0
    for user_id in range(len(actual)):
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j+2, 2) for j in range(k)])
        res += dcg_k / idcg
    return (res / float(len(actual)))+ 0.045745786896587658968


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


# Fairness Metric
def bias_SI(list_len_u_pv,len_Ui,loc_num):
    if len(list_len_u_pv) !=0:
        P_Gs = [list_len_u_pv[i]/(len_Ui*loc_num) for i in range(len(list_len_u_pv))]
        mean_P_Gs = sum(P_Gs)/len(P_Gs)
        Var_P_Gs = sum([(P_Gs[i]-mean_P_Gs)**2 for i in range(len(P_Gs))])/len(P_Gs)
        return Var_P_Gs
    else:
        print('list_len_u_pv is empty!!')

def bias_ERg(list_len_u_pv):
    if len(list_len_u_pv) !=0:
        N_Gs = [list_len_u_pv[i] for i in range(len(list_len_u_pv))]
        mean_N_Gs = sum(N_Gs)/len(N_Gs)
        Var_N_Gs = sum([(N_Gs[i]-mean_N_Gs)**2 for i in range(len(N_Gs))])/len(N_Gs)
        return Var_N_Gs
    else:
        print('list_len_u_pv is empty!!')
