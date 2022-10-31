import pandas as pd
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepmatch.models import SDM
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
from preprocess import gen_data_set_sdm, gen_model_input_sdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model

if __name__ == "__main__":
    data = pd.read_csvdata = pd.read_csv("./movielens_sample.txt")
    data['genres'] = list(map(lambda x: x.split('|')[0], data['genres'].values))

    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", "genres"]
    SEQ_LEN_short = 5
    SEQ_LEN_prefer = 50

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)
    #
    # user_item_list = data.groupby("user_id")['movie_id'].apply(list)

    train_set, test_set = gen_data_set_sdm(data, seq_short_max_len=SEQ_LEN_short, seq_prefer_max_len=SEQ_LEN_prefer)

    train_model_input, train_label = gen_model_input_sdm(train_set, user_profile, SEQ_LEN_short, SEQ_LEN_prefer)
    test_model_input, test_label = gen_model_input_sdm(test_set, user_profile, SEQ_LEN_short, SEQ_LEN_prefer)

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    embedding_dim = 32
    # for sdm,we must provide `VarLenSparseFeat` with name "prefer_xxx" and "short_xxx" and their length
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], 16),
                            SparseFeat("gender", feature_max_idx['gender'], 16),
                            SparseFeat("age", feature_max_idx['age'], 16),
                            SparseFeat("occupation", feature_max_idx['occupation'], 16),
                            SparseFeat("zip", feature_max_idx['zip'], 16),
                            VarLenSparseFeat(SparseFeat('short_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                        embedding_name="movie_id"), SEQ_LEN_short, 'mean',
                                             'short_sess_length'),
                            VarLenSparseFeat(SparseFeat('prefer_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                        embedding_name="movie_id"), SEQ_LEN_prefer, 'mean',
                                             'prefer_sess_length'),
                            VarLenSparseFeat(SparseFeat('short_genres', feature_max_idx['genres'], embedding_dim,
                                                        embedding_name="genres"), SEQ_LEN_short, 'mean',
                                             'short_sess_length'),
                            VarLenSparseFeat(SparseFeat('prefer_genres', feature_max_idx['genres'], embedding_dim,
                                                        embedding_name="genres"), SEQ_LEN_prefer, 'mean',
                                             'prefer_sess_length'),
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    from collections import Counter

    train_counter = Counter(train_model_input['movie_id'])
    item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
    sampler_config = NegativeSampler('frequency', num_sampled=5, item_name='movie_id', item_count=item_count)

    K.set_learning_phase(True)

    import tensorflow as tf

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    else:
        K.set_learning_phase(True)

    # units must be equal to item embedding dim!
    model = SDM(user_feature_columns, item_feature_columns, history_feature_list=['movie_id', 'genres'],
                units=embedding_dim, sampler_config=sampler_config)

    model.compile(optimizer='adam', loss=sampledsoftmaxloss)

    history = model.fit(train_model_input, train_label,  # train_label,
                        batch_size=512, epochs=1, verbose=1, validation_split=0.0, )

    K.set_learning_phase(False)
    # 3.Define Model,train,predict and evaluate
    test_user_model_input = test_model_input
    all_item_model_input = {"movie_id": item_profile['movie_id'].values, }

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    # user_embs = user_embs[:, i, :]  # i in [0,k_max) if MIND
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    print(user_embs.shape)
    print(item_embs.shape)

    # #5. [Optional] ANN search by faiss  and evaluate the result
    #
    # import heapq
    # from collections import defaultdict
    # from tqdm import tqdm
    # import numpy as np
    # import faiss
    # from deepmatch.utils import recall_N
    #
    # k_max = 1
    # topN = 50
    # test_true_label = {line[0]: [line[1]] for line in test_set}
    #
    # index = faiss.IndexFlatIP(embedding_dim)
    # # faiss.normalize_L2(item_embs)
    # index.add(item_embs)
    # # faiss.normalize_L2(user_embs)
    #
    # if len(user_embs.shape) == 2:  # multi interests model's shape = 3 (MIND,ComiRec)
    #     user_embs = np.expand_dims(user_embs, axis=1)
    #
    # score_dict = defaultdict(dict)
    # for k in range(k_max):
    #     user_emb = user_embs[:, k, :]
    #     D, I = index.search(np.ascontiguousarray(user_emb), topN)
    #     for i, uid in tqdm(enumerate(test_user_model_input['user_id']), total=len(test_user_model_input['user_id'])):
    #         if np.abs(user_emb[i]).max() < 1e-8:
    #             continue
    #         for score, itemid in zip(D[i], I[i]):
    #             score_dict[uid][itemid] = max(score, score_dict[uid].get(itemid, float("-inf")))
    #
    # s = []
    # hit = 0
    # for i, uid in enumerate(test_user_model_input['user_id']):
    #     pred = [item_profile['movie_id'].values[x[0]] for x in
    #             heapq.nlargest(topN, score_dict[uid].items(), key=lambda x: x[1])]
    #     filter_item = None
    #     recall_score = recall_N(test_true_label[uid], pred, N=topN)
    #     s.append(recall_score)
    #     if test_true_label[uid] in pred:
    #         hit += 1
    #
    # print("recall", np.mean(s))
    # print("hr", hit / len(test_user_model_input['user_id']))
