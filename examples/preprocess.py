import random
import numpy as np
from tqdm import tqdm
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def gen_data_set(data, negsample=0):
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i * negsample + negi], 0, len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set


def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label


def gen_model_input_gru4rec(data, batch_size, session_key, item_key, time_key):
    """
    Implement session-parallel mini-batches in 'session-based recommendations with recurrent neural networks'
    section 3.1.1.

    """
    data.sort_values([session_key, time_key], inplace=True)

    click_offsets = np.zeros(data[session_key].nunique() + 1, dtype=np.int32)
    # group & sort the df by session_key and get the offset values
    click_offsets[1:] = data.groupby(session_key).size().cumsum()

    session_idx_arr = np.arange(data[session_key].nunique())

    iters = np.arange(batch_size)
    maxiter = iters.max()
    start = click_offsets[session_idx_arr[iters]]
    end = click_offsets[session_idx_arr[iters] + 1]
    mask = []  # indicator for the sessions to be terminated
    finished = False

    while not finished:
        minlen = (end - start).min()
        # Item indices (for embedding) for clicks where the first sessions start
        idx_target = data[item_key].values[start]
        for i in range(minlen - 1):
            # Build inputs & targets
            idx_input = idx_target
            idx_target = data[item_key].values[start + i + 1]
            inp = idx_input
            target = idx_target
            yield inp, target, mask

        # click indices where a particular session meets second-to-last element
        start = start + (minlen - 1)
        # see if how many sessions should terminate
        mask = np.arange(len(iters))[(end - start) <= 1]
        done_sessions_counter = len(mask)
        for idx in mask:
            maxiter += 1
            if maxiter >= len(click_offsets) - 1:
                finished = True
                break
            # update the next starting/ending point
            iters[idx] = maxiter
            start[idx] = click_offsets[session_idx_arr[maxiter]]
            end[idx] = click_offsets[session_idx_arr[maxiter] + 1]
