import numpy as np
import pandas as pd

from deepctr.inputs import SparseFeat
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K

from deepmatch.utils import recall_N
from deepmatch.models.gru4rec import GRU4REC, top1, bpr
from preprocess import gen_model_input_gru4rec, gen_data_set

if __name__ == "__main__":
    debug = True
    if debug:
        data = pd.read_csvdata = pd.read_csv("./movielens_sample.txt")[['user_id', 'movie_id', 'timestamp']]
        batch_size = 3
    else:
        data_path = "./"
        unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
        user = pd.read_csv(data_path + 'ml-1m/users.dat', sep='::', header=None, names=unames)
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(data_path + 'ml-1m/ratings.dat', sep='::', header=None, names=rnames)
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_csv(data_path + 'ml-1m/movies.dat', sep='::', header=None, names=mnames)
        data = pd.merge(pd.merge(ratings, movies), user)[['user_id', 'movie_id', 'timestamp']]
        batch_size = 512

    features = ['user_id', 'movie_id']
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    data["rank"] = data.groupby("user_id")["timestamp"].rank("first", ascending=False)
    test_set = data.loc[data['rank'] <= 2,]
    train_set = data.loc[data['rank'] >= 2]

    epochs = 3
    embedding_dim = 128
    gru_units = (128,)
    n_classes = feature_max_idx['movie_id']
    loss_fn = 'CrossEntropy'

    test_loader = gen_model_input_gru4rec(test_set, batch_size, 'user_id', 'movie_id', 'timestamp')
    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    K.set_learning_phase(True)
    import tensorflow as tf

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    model = GRU4REC(item_feature_columns, n_classes, gru_units, batch_size)

    if loss_fn == 'CrossEntropy':
        model.compile(optimizer="adam", loss='sparse_categorical_crossentropy')
    elif loss_fn == 'TOP1':
        model.compile(optimizer="adam", loss=top1)
    elif loss_fn == 'BPR':
        model.compile(optimizer="adam", loss=bpr)

    model.summary()

    for epoch in range(epochs):
        step = 0
        train_loader = gen_model_input_gru4rec(train_set, batch_size, 'user_id', 'movie_id', 'timestamp')
        for feat, target, mask in train_loader:
            real_mask = np.ones((batch_size, 1))
            for elt in mask:
                real_mask[elt, :] = 0

            for i in range(len(gru_units)):
                hidden_states = K.get_value(model.get_layer('gru_{}'.format(str(i))).states[0])
                hidden_states = np.multiply(real_mask, hidden_states)
                hidden_states = np.array(hidden_states, dtype=np.float32)
                model.get_layer('gru_{}'.format(str(i))).reset_states(hidden_states)

            feat = np.array(feat).reshape((-1, 1))
            target = np.array(target).reshape((-1, 1))

            tr_loss = model.train_on_batch(feat, target)
            if step % 500 == 0:
                print(step)
                print(tr_loss)
            step += 1

    # s = []
    # hit = 0
    # total_sample = 0
    # n = 50
    # for feat, target, mask in test_loader:
    #     feat = np.array(feat).reshape((-1, 1))
    #     target = np.array(target).reshape((-1, 1))
    #     pred = model.predict(feat, batch_size=batch_size)
    #     pred = np.array(pred).argsort()[:, ::-1][:, :n]
    #
    #     for i in range(len(pred)):
    #         s.append(recall_N(target[i], pred[i], n))
    #         if target[i] in pred[i]:
    #             hit += 1
    #         total_sample += 1
    # print("recall", np.mean(s))
    # print("hit rate", hit / total_sample)
