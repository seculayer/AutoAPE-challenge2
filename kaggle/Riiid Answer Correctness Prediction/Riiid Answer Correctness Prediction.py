%%time
import pandas as pd
import numpy as np

print('Starting...')
train_df = pd.read_csv('input/riiid-test-answer-prediction/train.csv',
                       usecols=['timestamp',
                                'user_id',
                                'content_id',
                                'content_type_id',
                                'task_container_id',
                                'user_answer',
                                'prior_question_elapsed_time',
                                'prior_question_had_explanation',
                                'answered_correctly',],  #low_memory=False, #nrows=10**5,
                       dtype={'timestamp': 'int64',
                              'user_id': 'int32',
                              'content_id': 'int16',
                              'content_type_id': 'int8',
                              'task_container_id': 'int16',
                              'user_answer': 'int8',
                              'prior_question_elapsed_time': 'float32',
                              'prior_question_had_explanation': 'boolean',
                              'answered_correctly': 'int8',
                             }
                      )
print('Sorting...')
print('Filling...')
train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].astype(np.float).fillna(-1).astype(np.int8)
print('Filling...')
train_df['prior_question_elapsed_time'] = train_df['prior_question_elapsed_time'].fillna(-1000)
print(train_df.head())

print('Done')

%%time

import gc

gc.collect()


def compose_single_user_data(r):
    timestamps = r['timestamp'].values
    prev_timestamps = np.pad(timestamps[:-1], (1, 0), 'constant', constant_values=0).astype(np.int64)

    content_ids = r['content_id'].values
    content_type_ids = r['content_type_id'].values

    is_question = (content_type_ids == 0)
    question_timestamps = timestamps[is_question]
    question_prev_timestamps = prev_timestamps[is_question]

    question_ids = content_ids[is_question]
    question_task_container_ids = r['task_container_id'].values[is_question]
    user_answers = r['user_answer'].values[is_question]
    prior_question_elapsed_times = r['prior_question_elapsed_time'].values[is_question]
    prior_question_had_explanation_idxs = r['prior_question_had_explanation'].values[is_question] + 1
    answered_correctly_idxs = r['answered_correctly'].values[is_question] + 1

    prev_was_lecture = np.pad(content_type_ids[:-1], (1, 0), 'constant', constant_values=0).astype(np.int8)
    prev_prev_timestamps = np.pad(prev_timestamps[:-1], (1, 0), 'constant', constant_values=0).astype(np.int64)
    question_prev_was_lecture = prev_was_lecture[is_question]
    question_prev_prev_timestamps = prev_prev_timestamps[is_question]

    d = {'question_timestamps': question_timestamps,
         'question_prev_timestamps': question_prev_timestamps,
         'question_ids': question_ids,
         'question_task_container_ids': question_task_container_ids,
         'user_answers': user_answers,
         'prior_question_elapsed_times': prior_question_elapsed_times,
         'prior_question_had_explanation_idxs': prior_question_had_explanation_idxs,
         'answered_correctly_idxs': answered_correctly_idxs,

         'question_prev_was_lecture': question_prev_was_lecture,
         'question_prev_prev_timestamps': question_prev_prev_timestamps,
         }
    return d


print('Grouping...')
train_df = train_df.groupby('user_id')
print('Applying...')
train_df = train_df.apply(compose_single_user_data)
print('Done')

gc.collect()

pd.set_option('display.max_rows',100)
train_df.head(10)

user_idxs_by_ids = dict(zip(train_df.index, range(1, len(train_df) + 1)))
user_ids_by_idxs = np.array([-1] + list(train_df.index), dtype=np.int32)
user_data_by_idxs = [None] + list(train_df)
del train_df
user_data_questions_counts_by_idxs = [len(user_data['question_ids']) if user_data is not None else -1 for user_data in user_data_by_idxs]

user_id = 115
user_idx = user_idxs_by_ids[user_id]
print(f'user_id={user_id} user_idx={user_idx} user_ids_by_idxs[user_idxs_by_ids[user_id]]={user_ids_by_idxs[user_idxs_by_ids[user_id]]}')
user_data = user_data_by_idxs[user_idx]
print(f'user_data={str(user_data)[:1000]}')

questions_df = pd.read_csv('input/riiid-test-answer-prediction/questions.csv',
    dtype={'question_id': 'int16', 'bundle_id': 'int32', 'correct_answer': 'int8', 'part': 'int8', 'tags': 'object'})
questions_df.fillna('', inplace=True)
def extract_tags_list(x):
    tag_idxs = [0, 0, 0, 0, 0, 0]
    if type(x) is not str:
        print('not str:', type(x), x)
    else:
        if x != '':
            for i, part in enumerate(x.split(' ')):
                tag = int(part)
                if tag < 0 or tag > 187:
                    print('bad:', tag, x)
                tag_idxs[i] = tag + 1

    return tag_idxs
questions_df['tags_list'] = questions_df['tags'].apply(extract_tags_list)
questions_df.head(10)

print('unique question_ids:', len(questions_df['question_id'].unique()))
min_question_id = questions_df['question_id'].min()
print('min question_id:', min_question_id)
max_question_id = questions_df['question_id'].max()
print('max question_id:', max_question_id)

print('unique bundle_ids:', len(questions_df['bundle_id'].unique()))
print('min bundle_id:', questions_df['bundle_id'].min())
print('max bundle_id:', questions_df['bundle_id'].max())
print('unique correct_answers:', len(questions_df['correct_answer'].unique()))
print('min correct_answer:', questions_df['correct_answer'].min())
print('max correct_answer:', questions_df['correct_answer'].max())
print('unique parts:', len(questions_df['part'].unique()))
print('min part:', questions_df['part'].min())
print('max part:', questions_df['part'].max())

questions = questions_df.to_dict()
question_idxs_by_ids_dict = {}
question_idxs_by_ids = np.zeros((max_question_id+1,), dtype=np.int16)
for i, question_id in questions['question_id'].items():
    question_idxs_by_ids_dict[question_id] = i + 1
    question_idxs_by_ids[question_id] = i + 1
question_ids_by_idxs = np.array([-1] + list(questions['question_id']), dtype=np.int16)
question_bundle_ids_by_idxs = np.array([-1] + list(questions['bundle_id'].values()), dtype=np.int32)
question_correct_answers_by_idxs = np.array([-1] + list(questions['correct_answer'].values()), dtype=np.int8)

del questions
del questions_df
gc.collect()

question_id = 13522
question_idx = question_idxs_by_ids[question_id]
print(f'question_id={question_id} question_idx={question_idx} question_ids_by_idxs[question_idxs_by_ids[question_id]]={question_ids_by_idxs[question_idxs_by_ids[question_id]]}')


nonzero_user_idxs = np.arange(1, len(user_ids_by_idxs), dtype=np.int32)
print(f'nonzero_user_idxs: {len(nonzero_user_idxs)}')

test_set_fraction = 0.1
test_set_user_idxs = nonzero_user_idxs.copy()
np.random.shuffle(test_set_user_idxs)
test_set_user_idxs = np.array(sorted(test_set_user_idxs[:int(len(test_set_user_idxs) * test_set_fraction * 2)].tolist()))
test_set_user_idxs_set = set(test_set_user_idxs)
print(f'test_set_user_idxs: {len(test_set_user_idxs)} {list(test_set_user_idxs)[:100]}')

max_user_data_questions_count = 0
for user_idx in nonzero_user_idxs:
    user_data_questions_count = user_data_questions_counts_by_idxs[user_idx]
    if user_data_questions_count > max_user_data_questions_count:
        max_user_data_questions_count = user_data_questions_count
print(f'max_user_data_questions_count: {max_user_data_questions_count}')

def sample_test(max_seq_len):
    test_set_start_positions_by_user_idxs = [-1]
    test_user_idxs = []
    test_start_poss = []
    test_lengths = []
    test_roc_auc_start_poss = []
    for user_idx in nonzero_user_idxs:
        user_data_questions_count = user_data_questions_counts_by_idxs[user_idx]
        if user_idx in test_set_user_idxs_set:
            test_set_start_position_by_user_idx = np.random.randint(low=0, high=user_data_questions_count)
        else:
            test_set_start_position_by_user_idx = user_data_questions_count
        test_set_start_positions_by_user_idxs.append(test_set_start_position_by_user_idx)

        if user_idx in test_set_user_idxs_set:
            excess = (user_data_questions_count - test_set_start_position_by_user_idx) % max_seq_len
            if excess > 0:
                test_roc_auc_start_pos = test_set_start_position_by_user_idx
                start_pos = test_roc_auc_start_pos + excess - max_seq_len
                if start_pos < 0:
                    length = max_seq_len - (-start_pos)
                    start_pos = 0
                else:
                    length = max_seq_len
                test_user_idxs.append(user_idx)
                test_start_poss.append(start_pos)
                test_roc_auc_start_poss.append(test_roc_auc_start_pos)
                assert length > 0
                test_lengths.append(length)
            while start_pos < user_data_questions_count:
                length = min(user_data_questions_count - start_pos, max_seq_len)
                test_user_idxs.append(user_idx)
                test_start_poss.append(start_pos)
                test_roc_auc_start_poss.append(start_pos)
                assert length > 0
                test_lengths.append(length)
                start_pos += length
    return test_set_start_positions_by_user_idxs, test_user_idxs, test_start_poss, test_lengths, test_roc_auc_start_poss
test_set_start_positions_by_user_idxs, test_user_idxs, test_start_poss, test_lengths, test_roc_auc_start_poss = sample_test(200)
print(f'test_user_idxs: {len(test_user_idxs)} test_start_poss: {len(test_start_poss)} test_roc_auc_start_poss: {len(test_roc_auc_start_poss)} test_lengths: {len(test_lengths)}')

def sample_train(max_seq_len):
    train_user_idxs = []
    train_start_poss = []
    train_lengths = []
    train_samples_counts_by_user_idxs = [-1]
    for user_idx in nonzero_user_idxs:
        user_data_questions_count = user_data_questions_counts_by_idxs[user_idx]
        test_set_start_position_by_user_idx = test_set_start_positions_by_user_idxs[user_idx]

        start_pos = 0
        train_samples_count_by_user_idx = 0
        small_amount_of_data_for_user = (test_set_start_position_by_user_idx <= max_seq_len)
        while start_pos < test_set_start_position_by_user_idx:
            if (start_pos == 0):# and (not small_amount_of_data_for_user):
                high = min(test_set_start_position_by_user_idx, max_seq_len)
                if high == 1:
                    length = 1
                else:
                    length = np.random.randint(low=1, high=high)
            else:
                length = min(test_set_start_position_by_user_idx - start_pos, max_seq_len)
            if (length >= max_seq_len * 0.75) or (np.random.randint(low=0, high=100) >= 90):
                train_samples_count_by_user_idx += 1
                train_user_idxs.append(user_idx)
                train_start_poss.append(start_pos)
                assert length > 0
                train_lengths.append(length)
            start_pos += length
        train_samples_counts_by_user_idxs.append(train_samples_count_by_user_idx)
    train_indexes = np.arange(len(train_lengths))
    np.random.shuffle(train_indexes)
    return train_user_idxs, train_start_poss, train_lengths, train_indexes, train_samples_counts_by_user_idxs


train_user_idxs, train_start_poss, train_lengths, train_indexes, train_samples_counts_by_user_idxs = sample_train(200)
users_with_much_data_count = 0
for train_samples_count_by_user_idx in train_samples_counts_by_user_idxs:
    if train_samples_count_by_user_idx >= 5:
        users_with_much_data_count += 1
print(f'train_user_idxs: {len(train_user_idxs)} train_start_poss: {len(train_start_poss)} train_lengths: {len(train_lengths)} train_indexes: {len(train_indexes)} users_with_much_data_count: {users_with_much_data_count}')
del train_user_idxs
del train_start_poss
del train_lengths
del train_indexes
del train_samples_counts_by_user_idxs

del test_set_start_positions_by_user_idxs
del test_user_idxs
del test_start_poss
del test_lengths
del test_roc_auc_start_poss

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

train_samples_counts_by_user_idxs = None


class RiiidDataset(Dataset):
    def __init__(self, is_train_set=True, max_seq_len=128):
        super(RiiidDataset, self).__init__()
        self.is_train_set = is_train_set
        self.max_seq_len = max_seq_len

    def prepare(self):
        if self.is_train_set:
            global train_samples_counts_by_user_idxs
            self.set_user_idxs, self.set_start_poss, self.set_lengths, self.set_indexes, train_samples_counts_by_user_idxs_ = sample_train(
                self.max_seq_len)
            if train_samples_counts_by_user_idxs is None:
                train_samples_counts_by_user_idxs = train_samples_counts_by_user_idxs_
            self.set_roc_auc_start_poss = self.set_start_poss
        else:
            self.set_user_idxs, self.set_start_poss, self.set_lengths, self.set_roc_auc_start_poss = test_user_idxs, test_start_poss, test_lengths, test_roc_auc_start_poss

    def __len__(self):
        samples_count = len(self.set_lengths)  # //500
        return samples_count

    def __getitem__(self, index):
        if index < 0:
            user_idx = -index
            user_data = user_data_by_idxs[user_idx]
            seq_len = len(user_data['question_ids'])
            if seq_len >= self.max_seq_len:
                start_pos = seq_len - self.max_seq_len
                seq_len = self.max_seq_len
            else:
                start_pos = 0
            set_roc_auc_start_poss = start_pos

            user_data_test_start_pos = 0
        else:
            if self.is_train_set:
                index = self.set_indexes[index]
            user_idx = self.set_user_idxs[index]
            user_data = user_data_by_idxs[user_idx]
            user_data_test_start_pos = test_set_start_positions_by_user_idxs[user_idx]
            seq_len = self.set_lengths[index]
            start_pos = self.set_start_poss[index]
            set_roc_auc_start_poss = self.set_roc_auc_start_poss[index]

        user_data_seq = user_data['question_timestamps']
        question_timestamps = np.zeros(self.max_seq_len, dtype=user_data_seq.dtype)
        question_timestamps_without_padding = user_data_seq[start_pos:start_pos + seq_len]
        question_timestamps[:seq_len] = question_timestamps_without_padding

        user_data_seq = user_data['question_prev_timestamps']
        question_from_prev_timestamps_idxs = np.zeros(self.max_seq_len, dtype=np.int8)
        question_prev_timestamps_without_padding = user_data_seq[start_pos:start_pos + seq_len]
        question_from_prev_timestamps_idxs[:seq_len] = np.minimum(np.log(
            np.maximum(question_timestamps_without_padding - question_prev_timestamps_without_padding,
                       0) / 1000.0 + 2.0) / np.log(1.5), 31).astype(np.int8)
        if start_pos == 0:
            question_from_prev_timestamps_idxs[0] = 0

        user_data_seq = user_data['question_ids']
        question_idxs = np.zeros(self.max_seq_len, dtype=np.int32)
        question_idxs_without_padding = question_idxs_by_ids[user_data_seq[start_pos:start_pos + seq_len]]
        question_idxs[:seq_len] = question_idxs_without_padding

        user_data_seq = user_data['question_prev_was_lecture']

        question_prev_was_lecture_without_padding = user_data_seq[start_pos:start_pos + seq_len]
        user_data_seq = user_data['question_prev_prev_timestamps']
        question_prev_prev_timestamps_without_padding = user_data_seq[start_pos:start_pos + seq_len]
        question_pre_lecture_length_idxs_without_padding = np.minimum(np.log(
            np.maximum(question_prev_timestamps_without_padding - question_prev_prev_timestamps_without_padding,
                       0) / 1000.0 + 2.0) / np.log(1.5), 31).astype(np.int8)
        question_pre_lecture_length_idxs_without_padding = question_pre_lecture_length_idxs_without_padding * question_prev_was_lecture_without_padding
        question_pre_lecture_length_idxs = np.zeros(self.max_seq_len, dtype=np.int8)
        question_pre_lecture_length_idxs[:seq_len] = question_pre_lecture_length_idxs_without_padding
        if start_pos == 0:
            question_pre_lecture_length_idxs[0:2] = 0
        elif start_pos == 1:
            question_pre_lecture_length_idxs[0:1] = 0

        user_data_seq = user_data['question_task_container_ids']
        question_task_container_ids = np.zeros(self.max_seq_len, dtype=user_data_seq.dtype)
        question_task_container_ids_without_padding = user_data_seq[start_pos:start_pos + seq_len]
        is_bundle_id_changed = (question_task_container_ids_without_padding != np.pad(
            question_task_container_ids_without_padding[:-1], (1, 0), 'constant', constant_values=-1)).astype(np.int32)
        local_bundle_ids_without_padding = np.cumsum(is_bundle_id_changed)
        local_bundle_ids = np.zeros(self.max_seq_len, dtype=np.int32)
        local_bundle_ids[:seq_len] = local_bundle_ids_without_padding
        last_bundle_mask = np.zeros(self.max_seq_len, dtype=np.int8)
        last_bundle_id = local_bundle_ids_without_padding[-1]
        k = len(local_bundle_ids_without_padding) - 1
        while (k >= 0) and (local_bundle_ids_without_padding[k] == last_bundle_id):
            last_bundle_mask[k] = 1
            k -= 1

        user_data_seq = user_data['prior_question_elapsed_times']
        prior_question_elapsed_times_idxs = np.zeros(self.max_seq_len, dtype=np.int8)
        prior_question_elapsed_times_idxs_without_padding = user_data_seq[start_pos:start_pos + seq_len]
        prior_question_elapsed_times_idxs_without_padding = np.power(
            np.minimum(prior_question_elapsed_times_idxs_without_padding // 1000, 300) + 1, 0.726).astype(np.int8)
        prior_question_elapsed_times_idxs[:seq_len] = prior_question_elapsed_times_idxs_without_padding
        bundle_elapsed_times_idxs = np.zeros(self.max_seq_len + 2, dtype=np.int8)
        bundle_elapsed_times_idxs[
            local_bundle_ids_without_padding - 1 + 1] = prior_question_elapsed_times_idxs_without_padding
        bundle_elapsed_times_idxs = bundle_elapsed_times_idxs[1:]

        # print(f'mm: {local_bundle_ids_without_padding.min()} {local_bundle_ids_without_padding.max()}')

        question_elapsed_times_idxs_without_padding = bundle_elapsed_times_idxs[local_bundle_ids_without_padding]
        post_question_elapsed_times_idxs = np.zeros(self.max_seq_len, dtype=np.int8)
        post_question_elapsed_times_idxs[:seq_len] = question_elapsed_times_idxs_without_padding

        user_data_seq = user_data['prior_question_had_explanation_idxs']
        prior_question_had_explanation_idxs = np.zeros(self.max_seq_len, dtype=np.int8)
        prior_question_had_explanation_idxs_without_padding = user_data_seq[start_pos:start_pos + seq_len]
        prior_question_had_explanation_idxs[:seq_len] = prior_question_had_explanation_idxs_without_padding
        bundle_has_explanation_idxs = np.zeros(self.max_seq_len + 2, dtype=np.int8)
        bundle_has_explanation_idxs[
            local_bundle_ids_without_padding - 1 + 1] = prior_question_had_explanation_idxs_without_padding
        bundle_has_explanation_idxs = bundle_has_explanation_idxs[1:]
        question_has_explanation_idxs_without_padding = bundle_has_explanation_idxs[local_bundle_ids_without_padding]
        post_question_has_explanation_idxs = np.zeros(self.max_seq_len, dtype=np.int8)
        post_question_has_explanation_idxs[:seq_len] = question_has_explanation_idxs_without_padding

        user_data_seq = user_data['user_answers']
        user_answers_idxs = np.zeros(self.max_seq_len, dtype=np.int8)
        user_answers_idxs[:seq_len] = user_data_seq[start_pos:start_pos + seq_len] + 1

        correct_answers_idxs = np.zeros(self.max_seq_len, dtype=np.int8)
        correct_answers_idxs[:seq_len] = (question_correct_answers_by_idxs + 1)[question_idxs_without_padding]

        user_data_seq = user_data['answered_correctly_idxs']
        answered_correctly_idxs = np.zeros(self.max_seq_len, dtype=user_data_seq.dtype)
        answered_correctly_idxs_without_padding = user_data_seq[start_pos:start_pos + seq_len]
        answered_correctly_idxs[:seq_len] = answered_correctly_idxs_without_padding

        prior_question_was_answered_correctly_idxs = np.zeros(self.max_seq_len, dtype=np.int8)
        prior_question_was_answered_correctly_idxs[1:seq_len] = answered_correctly_idxs_without_padding[:-1]

        mask = np.zeros(self.max_seq_len, dtype=np.int8)
        mask[:seq_len] = 1

        if self.is_train_set:
            participates_in_roc_auc = mask
        else:
            participates_in_roc_auc = np.zeros(self.max_seq_len, dtype=np.int8)
            if set_roc_auc_start_poss - start_pos < seq_len - 1:
                participates_in_roc_auc[
                set_roc_auc_start_poss - start_pos:seq_len - 1] = 1  ####### hardcoded a patch (different from train variant)

        return (last_bundle_mask,  # user_idxs_,
                question_timestamps,

                question_pre_lecture_length_idxs,
                question_idxs,

                local_bundle_ids, prior_question_elapsed_times_idxs, prior_question_had_explanation_idxs,
                question_from_prev_timestamps_idxs,
                prior_question_was_answered_correctly_idxs,
                post_question_elapsed_times_idxs, post_question_has_explanation_idxs,

                user_answers_idxs,

                correct_answers_idxs,
                answered_correctly_idxs, participates_in_roc_auc, mask)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


from typing import Optional

use_reformer_self_attn = False


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if use_reformer_self_attn:
            src2 = self.self_attn(src.transpose(0, 1).contiguous()
                                  # , attn_mask=src_mask, key_padding_mask=src_key_padding_mask
                                  )
            src = src + self.dropout1(src2.transpose(0, 1).contiguous())
        else:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoderOnlyModel(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, num_layers, max_seq_len, dropout=0.1, use_performer=True,
                 feature_type='sqr', compute_type='iter', on_gptln=True):
        super(TransformerEncoderOnlyModel, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.max_seq_len = max_seq_len
        self.model_type = 'Transformer'

        self.use_performer = use_performer
        self.feature_type = feature_type
        self.compute_type = compute_type
        self.on_gptln = on_gptln

        # user_idxs_cnt = len(nonzero_user_idxs) + 1
        # self.user_embedding = nn.Embedding(user_idxs_cnt, d_model)

        self.day_part_embedding = nn.Embedding(240, d_model)

        self.question_pre_lecture_length_idxs_embedding = nn.Embedding(32, d_model)

        question_idxs_cnt = len(question_ids_by_idxs)
        self.question_embedding = nn.Embedding(question_idxs_cnt, d_model)

        self.question_postfactumness_embedding = nn.Embedding(3, d_model)

        self.local_bundle_ids_embedding = nn.Embedding(max_seq_len + 1, d_model)
        self.prior_question_elapsed_times_idxs_embedding = nn.Embedding(64, d_model)
        self.prior_question_had_explanation_idxs_embedding = nn.Embedding(3, d_model)
        self.question_from_prev_timestamps_idxs_embedding = nn.Embedding(32, d_model)
        self.prior_question_was_answered_correctly_idxs_embedding = nn.Embedding(3, d_model)
        self.post_question_elapsed_times_idxs_embedding = nn.Embedding(64, d_model)
        self.post_question_has_explanation_idxs_embedding = nn.Embedding(3, d_model)

        self.question_was_answered_correctly_idxs_embedding = nn.Embedding(3, d_model)
        self.correct_answer_embedding = nn.Embedding(5, d_model)

        if self.use_performer:
            vocab_size = None
            self.transformer_encoder = slim_performer_model.SLiMPerformer(vocab_size, d_model,
                                                                          num_layers, dim_feedforward,
                                                                          nhead, feature_type,
                                                                          compute_type,
                                                                          on_gptln)
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu')
            self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        self.encoder_top_linear_correct_answer_idx = nn.Linear(d_model, 1)
        self.encoder_top_linear_user_answer_idx = nn.Linear(d_model, 5)

        self.init_weights()

    def generate_square_subsequent_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.00001

        self.day_part_embedding.weight.data.uniform_(-initrange, initrange)

        self.question_pre_lecture_length_idxs_embedding.weight.data.uniform_(-initrange, initrange)
        self.question_embedding.weight.data.uniform_(-initrange, initrange)

        self.question_postfactumness_embedding.weight.data.uniform_(-initrange, initrange)

        self.local_bundle_ids_embedding.weight.data.uniform_(-initrange, initrange)
        self.prior_question_elapsed_times_idxs_embedding.weight.data.uniform_(-initrange, initrange)
        self.prior_question_had_explanation_idxs_embedding.weight.data.uniform_(-initrange, initrange)
        self.question_from_prev_timestamps_idxs_embedding.weight.data.uniform_(-initrange, initrange)
        self.prior_question_was_answered_correctly_idxs_embedding.weight.data.uniform_(-initrange, initrange)
        self.post_question_elapsed_times_idxs_embedding.weight.data.uniform_(-initrange, initrange)
        self.post_question_has_explanation_idxs_embedding.weight.data.uniform_(-initrange, initrange)

        self.question_was_answered_correctly_idxs_embedding.weight.data.uniform_(-initrange, initrange)
        self.correct_answer_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                user_idxs,
                timestamps,

                # lecture_idxs,
                question_pre_lecture_length_idxs,
                question_idxs,

                local_bundle_ids,
                prior_question_elapsed_times_idxs,
                prior_question_had_explanation_idxs,
                question_from_prev_timestamps_idxs,
                prior_question_was_answered_correctly_idxs,
                post_question_elapsed_times_idxs,
                post_question_has_explanation_idxs,

                question_was_answered_correctly_idxs,
                correct_answers_idxs,

                mask,

                attn_mask):

        day_part_embedding = self.day_part_embedding(((timestamps % 86400000) * (240.0 / 86400000)).long())

        # lecture_embedding = self.lecture_embedding(lecture_idxs)
        question_pre_lecture_length_idxs_embedding = self.question_pre_lecture_length_idxs_embedding(
            question_pre_lecture_length_idxs)
        question_embedding = self.question_embedding(question_idxs)  # * math.sqrt(self.d_model)

        question_postfactumness_embedding_x2 = self.question_postfactumness_embedding(
            mask.unsqueeze(-1) * torch.tensor([1, 2]).to(device))

        local_bundle_ids_embedding = self.local_bundle_ids_embedding(local_bundle_ids)
        prior_question_elapsed_times_idxs_embedding = self.prior_question_elapsed_times_idxs_embedding(
            prior_question_elapsed_times_idxs)
        prior_question_had_explanation_idxs_embedding = self.prior_question_had_explanation_idxs_embedding(
            prior_question_had_explanation_idxs)
        question_from_prev_timestamps_idxs_embedding = self.question_from_prev_timestamps_idxs_embedding(
            question_from_prev_timestamps_idxs)
        prior_question_was_answered_correctly_idxs_embedding = self.prior_question_was_answered_correctly_idxs_embedding(
            prior_question_was_answered_correctly_idxs)
        post_question_elapsed_times_idxs_embedding = self.post_question_elapsed_times_idxs_embedding(
            post_question_elapsed_times_idxs)
        post_question_has_explanation_idxs_embedding = self.post_question_has_explanation_idxs_embedding(
            post_question_has_explanation_idxs)

        question_was_answered_correctly_idxs_embedding = self.question_was_answered_correctly_idxs_embedding(
            question_was_answered_correctly_idxs)
        correct_answer_embedding = self.correct_answer_embedding(correct_answers_idxs)

        question_related_embeddings = (  # user_embedding +
            # lecture_embedding +
                question_pre_lecture_length_idxs_embedding
                + question_embedding

                + day_part_embedding

                + local_bundle_ids_embedding

                + question_from_prev_timestamps_idxs_embedding

                + correct_answer_embedding

        )
        question_related_embeddings_x2 = question_related_embeddings.unsqueeze(-2) * torch.tensor([[1.], [1.]]).to(
            device)

        posts_x2 = (
                               post_question_elapsed_times_idxs_embedding + post_question_has_explanation_idxs_embedding + question_was_answered_correctly_idxs_embedding).unsqueeze(
            -2) * torch.tensor([[0.], [1.]]).to(device)

        encoder_input_x2 = question_postfactumness_embedding_x2 + question_related_embeddings_x2 + posts_x2
        encoder_input = encoder_input_x2.view(
            [encoder_input_x2.size()[0], encoder_input_x2.size()[1] * encoder_input_x2.size()[2],
             encoder_input_x2.size()[3]])

        if not self.use_performer:
            encoder_input = encoder_input.transpose(0, 1)
            output = self.transformer_encoder(encoder_input, attn_mask)  # , src_key_padding_mask=(1-mask).bool())
        else:
            output = self.transformer_encoder.full_forward(encoder_input)
        output0 = self.encoder_top_linear_correct_answer_idx(output)
        if not self.use_performer:
            output0 = output0.view([output0.size()[0] // 2, 2, output0.size()[1], output0.size()[2]])[:, 0, :, :]
            output0 = output0.transpose(0, 1).contiguous()
        else:
            output0 = output0.view([output0.size()[0], output0.size()[1] // 2, 2, output0.size()[2]])[:, :, 0, :]
        output1 = self.encoder_top_linear_user_answer_idx(output)
        if not self.use_performer:
            output1 = output1.view([output1.size()[0] // 2, 2, output1.size()[1], output1.size()[2]])[:, 0, :, :]
            output1 = output1.transpose(0, 1).contiguous()
        else:
            output1 = output1.view([output1.size()[0], output1.size()[1] // 2, 2, output1.size()[2]])[:, :, 0, :]
        return output0, output1


from datetime import datetime

from torch.optim.lr_scheduler import LambdaLR
import math


class CustomBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, **kwargs):
        loss = ((target - 1) * torch.log(output) + (2 - target) * torch.log(1.0 - output)) * (target != 0).float()

        loss = torch.neg(torch.mean(loss))

        return loss


WARMUP_EPOCHS = 1.0 / 4.0
epochs = 5
lr = 0.0005
dropout = 0.0
MAX_SEQ_LEN = 256
batch_size = 64
valid_batch_size = 384

model = TransformerEncoderOnlyModel(d_model=512, nhead=8, dim_feedforward=384, num_layers=6,
                                    max_seq_len=MAX_SEQ_LEN * 2, dropout=dropout, use_performer=False,
                                    feature_type='relu', compute_type='iter', on_gptln=True)
criteria = [CustomBCELoss().to(device),
            nn.CrossEntropyLoss(ignore_index=0, weight=torch.tensor([0, 0.25, 0.25, 0.25, 0.25, ])).to(device),
            ]

best_model_file_path = 'input/model4pth/prf_wue0.25_sl256_lr0.0005_e5_of7_trainauc0.8082896856958341.pth'
print('Loading Model:', best_model_file_path)
model = torch.load(best_model_file_path)
model.to(device)

test_set_start_positions_by_user_idxs, test_user_idxs, test_start_poss, test_lengths, test_roc_auc_start_poss = sample_test(
    MAX_SEQ_LEN)

valid_dataset = RiiidDataset(is_train_set=False, max_seq_len=MAX_SEQ_LEN)

print('MAX_SEQ_LEN:', MAX_SEQ_LEN, model.max_seq_len)
print('epochs:', epochs)
print('WARMUP_EPOCHS:', WARMUP_EPOCHS)
print('lr:', lr)
print('dropout:', dropout)
print('batch_size:', batch_size)
print('valid_batch_size:', valid_batch_size)
print('nhead:', model.nhead)
print(model)

gc.collect()


class Iter_Valid(object):
    def __init__(self, df, max_user=1000):
        df = df.reset_index(drop=True)
        self.df = df
        self.user_answer = df['user_answer'].astype(str).values
        self.answered_correctly = df['answered_correctly'].astype(str).values
        df['prior_group_responses'] = "[]"
        df['prior_group_answers_correct'] = "[]"
        self.sample_df = df[df['content_type_id'] == 0][['row_id']]
        self.sample_df['answered_correctly'] = 0
        self.len = len(df)
        self.user_id = df.user_id.values
        self.task_container_id = df.task_container_id.values
        self.content_type_id = df.content_type_id.values
        self.max_user = max_user
        self.current = 0
        self.pre_user_answer_list = []
        self.pre_answered_correctly_list = []

    def __iter__(self):
        return self

    def fix_df(self, user_answer_list, answered_correctly_list, pre_start):
        df = self.df[pre_start:self.current].copy()
        sample_df = self.sample_df[pre_start:self.current].copy()
        df.loc[pre_start, 'prior_group_responses'] = '[' + ",".join(self.pre_user_answer_list) + ']'
        df.loc[pre_start, 'prior_group_answers_correct'] = '[' + ",".join(self.pre_answered_correctly_list) + ']'
        print('lll', df.loc[pre_start, 'prior_group_answers_correct'])
        self.pre_user_answer_list = user_answer_list
        self.pre_answered_correctly_list = answered_correctly_list
        return df, sample_df

    def __next__(self):
        found_good = False

        while not found_good:
            added_user = set()
            pre_start = self.current
            pre_added_user = -1
            pre_task_container_id = -1

            user_answer_list = []
            answered_correctly_list = []
            result = None
            while self.current < self.len:
                crr_user_id = self.user_id[self.current]
                crr_task_container_id = self.task_container_id[self.current]
                crr_content_type_id = self.content_type_id[self.current]
                if crr_content_type_id == 1:
                    # no more than one task_container_id of "questions" from any single user
                    # so we only care for content_type_id == 0 to break loop
                    # user_answer_list.append(self.user_answer[self.current])
                    # answered_correctly_list.append(self.answered_correctly[self.current])
                    self.current += 1
                    continue
                if crr_user_id in added_user and ((crr_user_id != pre_added_user) or
                                                  (crr_task_container_id != pre_task_container_id)):
                    # known user(not prev user or differnt task container)
                    result = self.fix_df(user_answer_list, answered_correctly_list, pre_start)
                    break
                if len(added_user) == self.max_user:
                    if crr_user_id == pre_added_user and crr_task_container_id == pre_task_container_id:
                        user_answer_list.append(self.user_answer[self.current])
                        answered_correctly_list.append(self.answered_correctly[self.current])
                        self.current += 1
                        continue
                    else:
                        result = self.fix_df(user_answer_list, answered_correctly_list, pre_start)
                        break
                added_user.add(crr_user_id)
                pre_added_user = crr_user_id
                pre_task_container_id = crr_task_container_id
                user_answer_list.append(self.user_answer[self.current])
                answered_correctly_list.append(self.answered_correctly[self.current])
                self.current += 1
            if result is None:
                if pre_start < self.current:
                    result = self.fix_df(user_answer_list, answered_correctly_list, pre_start)
                else:
                    raise StopIteration()

            dff = result[0]
            if len(dff[dff['content_type_id'] == 0]) > 0:
                found_good = True
                return dff, result[1]
            else:
                found_good = False


import riiideducation

env = riiideducation.make_env()
iter_test = env.iter_test()

from tqdm.notebook import tqdm

user_data_by_idxs = list(user_data_by_idxs)
user_ids_by_idxs = list(user_ids_by_idxs)
# print(user_ids_by_idxs[-10:])

fake_dataset = RiiidDataset(is_train_set=False, max_seq_len=MAX_SEQ_LEN)
model.eval()


def process_batch(batch_as_item, batch_positions_of_interest, batch_id):
    item = batch_as_item

    losses = []
    losses_sum = 0.0
    num_corrects = 0
    num_total = 0
    labels = []
    outputs = []
    predictions = []

    with torch.no_grad():
        attn_mask = model.generate_square_subsequent_mask(seq_len=model.max_seq_len).to(device)

        user_idxs = item[0].to(device).long()

        timestamps = item[1].to(device).double()

        # lecture_idxs = item[2].to(device).long()
        question_pre_lecture_length_idxs = item[2].to(device).long()
        question_idxs = item[3].to(device).long()

        local_bundle_ids = item[4].to(device).long()
        prior_question_elapsed_times_idxs = item[5].to(device).long()
        prior_question_had_explanation_idxs = item[6].to(device).long()
        question_from_prev_timestamps_idxs = item[7].to(device).long()
        prior_question_was_answered_correctly_idxs = item[8].to(device).long()
        post_question_elapsed_times_idxs = item[9].to(device).long()
        post_question_has_explanation_idxs = item[10].to(device).long()

        label_user_answers_idxs = item[11].to(device).long()

        correct_answers_idxs = item[12].to(device).long()
        label_answered_correctly_idxs = item[-3].to(device).long()
        participates_in_roc_auc_long = item[-2]
        participates_in_roc_auc_long = participates_in_roc_auc_long.to(device).long()
        participates_in_roc_auc = participates_in_roc_auc_long.float()
        mask = item[-1].to(device).long()

        output0, output1 = model(user_idxs,
                                 timestamps,

                                 # lecture_idxs,
                                 question_pre_lecture_length_idxs,
                                 question_idxs,

                                 local_bundle_ids, prior_question_elapsed_times_idxs,
                                 prior_question_had_explanation_idxs,
                                 question_from_prev_timestamps_idxs,
                                 prior_question_was_answered_correctly_idxs,
                                 post_question_elapsed_times_idxs,
                                 post_question_has_explanation_idxs,

                                 label_answered_correctly_idxs,
                                 correct_answers_idxs,
                                 mask,
                                 attn_mask)
        output0 = output0.squeeze(-1)

        output0 = torch.sigmoid(output0)
        pred = (output0 >= 0.5).long()
        pred = pred + 1

        # print(f'len(batch_positions_of_interest)={len(batch_positions_of_interest)} output0.size()={output0.size()}')

        idx0 = torch.arange(len(batch_positions_of_interest)).to(device)

        # print(f'idx0.size()={idx0.size()}')

        idx1 = torch.tensor(batch_positions_of_interest).to(device)

        # print(f'idx1.size()={idx1.size()}')

        batch_predictions = output0[idx0, idx1]
        batch_predictions = batch_predictions.detach()
        batch_predictions = batch_predictions.cpu()
        batch_predictions = batch_predictions.numpy().astype(np.float64)
        loss0 = criteria[0](output0.view(-1), label_answered_correctly_idxs.float().view(-1))

        loss1 = criteria[1](output1.reshape(-1, 5), label_user_answers_idxs.view(-1))

        loss = (loss0 + loss1)  # / loss_normalizer

        loss_val = loss.detach().item()
        losses.append(loss_val)
        losses_sum += loss_val

        num_corrects += ((pred == label_answered_correctly_idxs) * participates_in_roc_auc).sum().item()
        num_total += participates_in_roc_auc.sum().item()

        auc_roc_applicable_positions_mask_flattened = (
                    participates_in_roc_auc * label_answered_correctly_idxs).bool().view(-1).cpu().numpy()
        label_answered_correctly_idxs_flattened = (label_answered_correctly_idxs - 1).view(-1).cpu().numpy()
        labels.extend(label_answered_correctly_idxs_flattened[auc_roc_applicable_positions_mask_flattened])

        output_flattened = output0.view(-1).data.cpu().numpy()
        outputs.extend(output_flattened[auc_roc_applicable_positions_mask_flattened])

    if batch_id % 100 == 0:
        if len(outputs) > 0:
            acc = num_corrects / max(0.000001, num_total)
            if np.max(labels) != np.min(labels):
                auc = roc_auc_score(labels, outputs)
            else:
                auc = 0.0
            loss = np.mean(losses)  # TODO: need direct mean, not mean of means
        else:
            acc = 0.0
            auc = 0.0
            loss = np.mean(losses)  # TODO: need direct mean, not mean of means
        # print(f'outputs', len(outputs), 'labels', len(labels))
        # print(f'labels', np.array(labels).min(), np.array(labels).max())
        # if len(outputs) > 0:
        #    print(f'outputs[0]', outputs[0], outputs[-1])
        #    print(f'labels[0]', labels[0], labels[-1])

        print(
            f'{datetime.now().isoformat()}: batch_id={batch_id} batch_len={len(batch)} loss={loss}, acc={acc}, auc={auc}')

    return batch_predictions


# batch_predictions = process_batch(batch, batch_positions_of_interest, 0)
# print(batch_predictions)


def update_prev_group(prior_group_answers_correct, prior_group_responses, prior_group_user_idxs):
    prior_group_answers_correct_idxs = np.array([int(s) + 1 for s in (
        prior_group_answers_correct[1:-1].replace(' ', '').split(',') if prior_group_answers_correct != '[]' else [])],
                                                dtype=np.int8)
    prior_group_responses = np.array([int(s) for s in (
        prior_group_responses[1:-1].replace(' ', '').split(',') if prior_group_responses != '[]' else [])],
                                     dtype=np.int8)
    assert len(prior_group_answers_correct_idxs) == len(prior_group_responses)
    assert len(prior_group_answers_correct_idxs) == len(
        prior_group_user_idxs), f'len(prior_group_answers_correct_idxs) == len(prior_group_user_idxs): {len(prior_group_answers_correct_idxs)} {len(prior_group_user_idxs)}'

    user_data_skips_by_idxs = np.zeros(len(user_data_by_idxs), dtype='int')
    for i in range(len(prior_group_responses) - 1, -1, -1):
        user_idx = prior_group_user_idxs[i]
        if user_idx != 0:
            skip = user_data_skips_by_idxs[user_idx]
            user_data_skips_by_idxs[user_idx] += 1
            # print(f'patching user_idx {user_idx}')
            # print(f'******************* prior_group_answers_correct_idxs {prior_group_answers_correct_idxs}')
            # print(f'******************* patching prior_group_answers_correct_idxs {prior_group_answers_correct_idxs}')
            user_data = user_data_by_idxs[user_idx]
            user_answers = user_data['user_answers']
            assert len(user_answers) > skip, f'len(user_answers) >= skip: {len(user_answers)} > {skip}'
            user_answers[-skip - 1] = prior_group_responses[i]

            answered_correctly_idxs = user_data['answered_correctly_idxs']
            assert len(
                answered_correctly_idxs) > skip, f'len(answered_correctly_idxs) > skip: {len(answered_correctly_idxs)} >= {skip}'
            answered_correctly_idxs[-skip - 1] = prior_group_answers_correct_idxs[i]


group_user_idxs = []
batch_id = 0
print(f'{datetime.now().isoformat()}: Starting dfs')
for (test_df, sample_prediction_df) in iter_test:

    dbg_cond = batch_id < 10

    if dbg_cond:
        print(
            f'{datetime.now().isoformat()}: Starting df of size {len(test_df)} ({len(test_df[test_df["content_type_id"] == 0])}q {len(test_df[test_df["content_type_id"] == 1])}l) sample_prediction_df: {sample_prediction_df}')

    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].astype(np.float).fillna(
        -1).astype(np.int8)
    test_df['prior_question_elapsed_time'] = test_df['prior_question_elapsed_time'].fillna(-1000)
    if dbg_cond:
        # print(f'df columns: {test_df.columns}')
        # print(f'df dtypes: {test_df.dtypes}')
        print(test_df.head(10))

    answered_correctly = [0.0 for l in range(len(test_df))]

    prior_group_user_idxs = group_user_idxs
    group_user_idxs = []

    prev_group_num = -1
    batch_samples = [[] for i in range(16)]
    batch_positions_to_df_positions = []
    batch_positions_of_interest = []

    for df_position, (group_num, row) in enumerate(test_df.iterrows()):
        # group_num, row_id, timestamp, user_id, content_id, content_type_id, task_container_id, prior_question_elapsed_time, prior_question_had_explanation, prior_group_answers_correct, prior_group_responses = row
        timestamp = row['timestamp']
        user_id = row['user_id']
        content_id = row['content_id']
        content_type_id = row['content_type_id']
        task_container_id = row['task_container_id']
        prior_question_elapsed_time = row['prior_question_elapsed_time']
        prior_question_had_explanation = row['prior_question_had_explanation']
        prior_group_answers_correct = row['prior_group_answers_correct']
        prior_group_responses = row['prior_group_responses']

        assert content_type_id == 0 or content_type_id == 1

        if dbg_cond:
            print('row, group_num:', row, group_num)

        # if group_num != prev_group_num:
        if df_position == 0:
            update_prev_group(prior_group_answers_correct, prior_group_responses, prior_group_user_idxs)

        if user_id in user_idxs_by_ids:
            user_idx = user_idxs_by_ids[user_id]
            user_data = user_data_by_idxs[user_idx]
        else:
            user_data = {
                'question_timestamps': np.array([], dtype=np.int64),
                'question_prev_timestamps': np.array([], dtype=np.int64),

                'question_ids': np.array([], dtype=np.int16),
                'question_task_container_ids': np.array([], dtype=np.int16),
                'user_answers': np.array([], dtype=np.int8),
                'prior_question_elapsed_times': np.array([], dtype=np.float32),
                'prior_question_had_explanation_idxs': np.array([], dtype=np.int8),
                'answered_correctly_idxs': np.array([], dtype=np.int8),

                'question_prev_was_lecture': np.array([], dtype=np.int8),
                'question_prev_prev_timestamps': np.array([], dtype=np.int64),
            }
            user_idx = len(user_data_by_idxs)
            user_data_by_idxs.append(user_data)
            user_idxs_by_ids[user_id] = user_idx
            user_data_questions_counts_by_idxs.append(0)
            user_ids_by_idxs.append(user_id)

        if content_type_id == 1:
            user_data['reminder__question_prev_lecture_timestamp'] = (timestamp)
            group_user_idxs.append(0)
        else:

            if 'reminder__question_prev_lecture_timestamp' in user_data:
                reminder__question_prev_lecture_timestamp = user_data['reminder__question_prev_lecture_timestamp']
                reminder__prev_was_lecture = 1
                del user_data['reminder__question_prev_lecture_timestamp']
            else:
                reminder__question_prev_lecture_timestamp = None
                reminder__prev_was_lecture = 0

            user_data_questions_counts_by_idxs[user_idx] += 1

            question_timestamps = user_data['question_timestamps']
            question_timestamps = np.append(question_timestamps, timestamp)
            if len(question_timestamps) > MAX_SEQ_LEN + 2:
                question_timestamps = question_timestamps[-MAX_SEQ_LEN - 2:]
                user_data_questions_counts_by_idxs[user_idx] = MAX_SEQ_LEN + 2
            user_data['question_timestamps'] = question_timestamps

            question_prev_timestamps = user_data['question_prev_timestamps']
            if reminder__question_prev_lecture_timestamp is not None:
                question_prev_timestamp = reminder__question_prev_lecture_timestamp
            else:
                if len(question_timestamps) >= 2:
                    question_prev_timestamp = question_timestamps[-2]
                else:
                    question_prev_timestamp = 0
            question_prev_timestamps = np.append(question_prev_timestamps, question_prev_timestamp)
            if len(question_prev_timestamps) > MAX_SEQ_LEN + 2:
                question_prev_timestamps = question_prev_timestamps[-MAX_SEQ_LEN - 2:]
            user_data['question_prev_timestamps'] = question_prev_timestamps

            question_ids = user_data['question_ids']
            question_ids = np.append(question_ids, content_id)
            if len(question_ids) > MAX_SEQ_LEN + 2:
                question_ids = question_ids[-MAX_SEQ_LEN - 2:]
            user_data['question_ids'] = question_ids

            question_task_container_ids = user_data['question_task_container_ids']
            question_task_container_ids = np.append(question_task_container_ids, task_container_id)
            if len(question_task_container_ids) > MAX_SEQ_LEN + 2:
                question_task_container_ids = question_task_container_ids[-MAX_SEQ_LEN - 2:]
            user_data['question_task_container_ids'] = question_task_container_ids

            user_answers = user_data['user_answers']
            user_answers = np.append(user_answers, 0)
            if len(user_answers) > MAX_SEQ_LEN + 2:
                user_answers = user_answers[-MAX_SEQ_LEN - 2:]
            user_data['user_answers'] = user_answers

            prior_question_elapsed_times = user_data['prior_question_elapsed_times']
            prior_question_elapsed_times = np.append(prior_question_elapsed_times, prior_question_elapsed_time)
            if len(prior_question_elapsed_times) > MAX_SEQ_LEN + 2:
                prior_question_elapsed_times = prior_question_elapsed_times[-MAX_SEQ_LEN - 2:]
            user_data['prior_question_elapsed_times'] = prior_question_elapsed_times

            prior_question_had_explanation_idxs = user_data['prior_question_had_explanation_idxs']
            prior_question_had_explanation_idxs = np.append(prior_question_had_explanation_idxs,
                                                            prior_question_had_explanation + 1)
            if len(prior_question_had_explanation_idxs) > MAX_SEQ_LEN + 2:
                prior_question_had_explanation_idxs = prior_question_had_explanation_idxs[-MAX_SEQ_LEN - 2:]
            user_data['prior_question_had_explanation_idxs'] = prior_question_had_explanation_idxs

            answered_correctly_idxs = user_data['answered_correctly_idxs']
            answered_correctly_idxs = np.append(answered_correctly_idxs, 0)
            if len(answered_correctly_idxs) > MAX_SEQ_LEN + 2:
                answered_correctly_idxs = answered_correctly_idxs[-MAX_SEQ_LEN - 2:]
            user_data['answered_correctly_idxs'] = answered_correctly_idxs

            question_prev_was_lecture = user_data['question_prev_was_lecture']
            question_prev_was_lecture = np.append(question_prev_was_lecture, reminder__prev_was_lecture)
            if len(question_prev_was_lecture) > MAX_SEQ_LEN + 2:
                question_prev_was_lecture = question_prev_was_lecture[-MAX_SEQ_LEN - 2:]
            user_data['question_prev_was_lecture'] = question_prev_was_lecture

            question_prev_prev_timestamps = user_data['question_prev_prev_timestamps']
            if len(question_prev_timestamps) >= 2:
                question_prev_prev_timestamp = question_prev_timestamps[-2]
            else:
                question_prev_prev_timestamp = 0
            question_prev_prev_timestamps = np.append(question_prev_prev_timestamps, question_prev_prev_timestamp)
            if len(question_prev_prev_timestamps) > MAX_SEQ_LEN + 2:
                question_prev_prev_timestamps = question_prev_prev_timestamps[-MAX_SEQ_LEN - 2:]
            user_data['question_prev_prev_timestamps'] = question_prev_prev_timestamps

            #         print(f'user_id={user_id} user_idx={user_idx} user_idxs_by_ids[user_id]={user_idxs_by_ids[user_id]} {len(user_ids_by_idxs)}')
            #         print(f'user_id={user_id} user_idx={user_idx} user_ids_by_idxs[user_idxs_by_ids[user_id]]={user_ids_by_idxs[user_idxs_by_ids[user_id]]}')
            #         print(f'user_data={str(user_data)[:1000]}')

            if dbg_cond:
                print('u_idx:', user_idx, batch_id)
            group_user_idxs.append(user_idx)
            sample = fake_dataset.__getitem__(-user_idx)
            mask = sample[-1]
            positions_of_interest = np.nonzero(mask)[0]
            if len(positions_of_interest) > 0:
                position_of_interest = positions_of_interest[-1]
            else:  # Actually this should never happen
                position_of_interest = 0

            batch_positions_to_df_positions.append(df_position)
            batch_positions_of_interest.append(position_of_interest)
            for i in range(len(sample)):
                batch_samples[i].append(sample[i])

        need_to_update_batch_id = False
        if (len(batch_samples[0]) == valid_batch_size) or (df_position == len(test_df) - 1):

            if len(batch_samples[0]) > 0:
                batch = [torch.tensor(batch_samples[i]) for i in range(len(batch_samples))]

                if dbg_cond:
                    print(f'{datetime.now().isoformat()}: Predicting batch {batch_id}')
                    for ii in range(16):
                        print(batch_samples[ii])
                batch_predictions = process_batch(batch, batch_positions_of_interest, batch_id)
                for batch_pos, batch_position_to_df_position in enumerate(batch_positions_to_df_positions):
                    answered_correctly[batch_position_to_df_position] = batch_predictions[batch_pos]
                # if dbg_cond:
                #    print(f'{datetime.now().isoformat()}: batch_predictions {batch_id}: {batch_predictions}')

                batch_samples = [[] for i in range(16)]
                batch_positions_to_df_positions = []
                batch_positions_of_interest = []
                need_to_update_batch_id = True

        if df_position == len(test_df) - 1:
            assert len(test_df) == len(answered_correctly)
            if dbg_cond:
                print(f'{datetime.now().isoformat()}: Sending df predictions')
            # test_df['answered_correctly'] = 0.5
            # if dbg_cond:
            #    print(f'dtypes: {test_df.dtypes}')
            test_df['answered_correctly'] = np.array(answered_correctly)
            #             if dbg_cond:
            #                 print(f'dtypes: {test_df.dtypes}')
            #                 print(f'dtype: {np.array(answered_correctly).dtype}')
            #                 print(f'arr: {np.array(answered_correctly)}')
            #                 print(f'{datetime.now().isoformat()}: pre - Sent df predictions: {test_df.loc[test_df["content_type_id"] == 0, ["row_id", "answered_correctly"]]}')
            env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

            # pbar.update(len(test_df))
            if dbg_cond:
                print(f'{datetime.now().isoformat()}: Sent df predictions')

        if need_to_update_batch_id:
            batch_id += 1

        # prev_group_num = group_num

    if dbg_cond:
        print(f'{datetime.now().isoformat()}: Finished df')
print(f'{datetime.now().isoformat()}: Done')













