import numpy as np
import random
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


def gen_data_set(data, seq_max_len=50, negsample=0):
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()
    item_id_genres_map = dict(zip(data['movie_id'].values, data['genres'].values))
    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        genres_list = hist['genres'].tolist()
        rating_list = hist['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            genres_hist = genres_list[:i]
            seq_len = min(i, seq_max_len)
            if i != len(pos_list) - 1:
                train_set.append((
                    reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
                    genres_list[i],
                    rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewerID, neg_list[i * negsample + negi], 0, hist[::-1][:seq_len], seq_len,
                                      genres_hist[::-1][:seq_len], item_id_genres_map[neg_list[i * negsample + negi]]))
            else:
                test_set.append((reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
                                 genres_list[i],
                                 rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


def gen_data_set_sdm(data, seq_short_max_len=5, seq_prefer_max_len=50):
    data.sort_values("timestamp", inplace=True)
    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        genres_list = hist['genres'].tolist()
        rating_list = hist['rating'].tolist()
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            genres_hist = genres_list[:i]
            seq_short_len = min(i, seq_short_max_len)
            seq_prefer_len = min(max(i - seq_short_len, 0), seq_prefer_max_len)
            if i != len(pos_list) - 1:
                train_set.append(
                    (reviewerID, pos_list[i], 1, hist[::-1][:seq_short_len][::-1],
                     hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], seq_short_len,
                     seq_prefer_len, genres_hist[::-1][:seq_short_len][::-1],
                     genres_hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], rating_list[i]))
            else:
                test_set.append(
                    (reviewerID, pos_list[i], 1, hist[::-1][:seq_short_len][::-1],
                     hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], seq_short_len,
                     seq_prefer_len, genres_hist[::-1][:seq_short_len][::-1],
                     genres_hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_iid = np.array([line[1] for line in train_set])
    train_label = np.array([line[2] for line in train_set])
    train_seq = [line[3] for line in train_set]
    train_hist_len = np.array([line[4] for line in train_set])
    train_seq_genres = np.array([line[5] for line in train_set])
    train_genres = np.array([line[6] for line in train_set])
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_seq_genres_pad = pad_sequences(train_seq_genres, maxlen=seq_max_len, padding='post', truncating='post',
                                         value=0)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_genres": train_seq_genres_pad,
                         "hist_len": train_hist_len, "genres": train_genres}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label


def gen_model_input_sdm(train_set, user_profile, seq_short_max_len, seq_prefer_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_iid = np.array([line[1] for line in train_set])
    train_label = np.array([line[2] for line in train_set])
    short_train_seq = [line[3] for line in train_set]
    prefer_train_seq = [line[4] for line in train_set]
    train_short_len = np.array([line[5] for line in train_set])
    train_prefer_len = np.array([line[6] for line in train_set])
    short_train_seq_genres = np.array([line[7] for line in train_set])
    prefer_train_seq_genres = np.array([line[8] for line in train_set])

    train_short_item_pad = pad_sequences(short_train_seq, maxlen=seq_short_max_len, padding='post', truncating='post',
                                         value=0)
    train_prefer_item_pad = pad_sequences(prefer_train_seq, maxlen=seq_prefer_max_len, padding='post',
                                          truncating='post',
                                          value=0)
    train_short_genres_pad = pad_sequences(short_train_seq_genres, maxlen=seq_short_max_len, padding='post',
                                           truncating='post',
                                           value=0)
    train_prefer_genres_pad = pad_sequences(prefer_train_seq_genres, maxlen=seq_prefer_max_len, padding='post',
                                            truncating='post',
                                            value=0)

    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "short_movie_id": train_short_item_pad,
                         "prefer_movie_id": train_prefer_item_pad,
                         "prefer_sess_length": train_prefer_len,
                         "short_sess_length": train_short_len, 'short_genres': train_short_genres_pad,
                         'prefer_genres': train_prefer_genres_pad}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label
