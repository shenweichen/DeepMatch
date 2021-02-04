"""
Author:
    Zichao Li, 2843656167@qq.com

Reference:
    Jing Li, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Tao Lian, and Jun Ma. 2017. Neural Attentive Session-based Recommendation. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (CIKM '17). Association for Computing Machinery, New York, NY, USA, 1419–1428.
"""

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Dropout
from deepctr.feature_column import build_input_features, create_embedding_matrix, varlen_embedding_lookup
from deepctr.layers.sequence import DynamicGRU
from deepctr.layers.utils import NoMask, concat_func
from ..layers.core import EmbeddingIndex
from ..layers import PoolingLayer, SampledSoftmaxLayer
from ..utils import get_item_embedding


def NARM(user_feature_columns, item_feature_columns, num_sampled=5, gru_hidden_units=(100,), emb_dropout_rate=0.25,
         output_dropout_rate=0.5, l2_reg_embedding=1e-6, seed=2021):
    """
    Instantiates the NARM Model architecture.

    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param num_sampled: int, the number of classes to randomly sample per batch.
    :param gru_hidden_units: list, the length of gru_hidden_units is equal to the number of gru layers.
    :param emb_dropout_rate: float in [0,1), this dropout layer is used after the user embedding layer.
    :param output_dropout_rate: float in [0,1), this dropout layer is used before the final user output.
    :param l2_reg_embedding: L2 regularizer strength applied to embedding vector.
    :param seed: int, to use as random seed.
    """
    if len(user_feature_columns) > 1:
        raise ValueError(
            "NARM only accept user behavior sequence as user feature.")
    if len(item_feature_columns) > 1:
        raise ValueError("Now NARM only support 1 item feature like item_id")

    # for user
    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())

    user_sess_length = user_features[user_feature_columns[0].length_name]

    # for item id
    item_feature_name = item_feature_columns[0].name
    item_vocabulary_size = item_feature_columns[0].vocabulary_size
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding, seed)

    # item embedding
    item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[item_feature_name])
    item_embedding_matrix = embedding_matrix_dict[item_feature_name]
    item_embedding_weight = NoMask()(item_embedding_matrix(item_index))
    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])

    # global encoder
    user_varlen_sparse_embedding_dict = varlen_embedding_lookup(embedding_matrix_dict, user_features,
                                                                user_feature_columns)
    user_varlen_sparse_embedding = user_varlen_sparse_embedding_dict[user_feature_columns[0].name]
    user_varlen_sparse_embedding = Dropout(rate=emb_dropout_rate)(user_varlen_sparse_embedding)

    user_gru_output = user_varlen_sparse_embedding
    for i in range(len(gru_hidden_units)):
        user_gru_output = DynamicGRU(num_units=gru_hidden_units[i])(
            [user_gru_output, user_sess_length])

    user_global_output = user_gru_output[:, -1, :]

    # local encoder
    user_gru_output_shape = user_gru_output.get_shape().as_list()
    q_1 = Dense(units=gru_hidden_units[-1], use_bias=False)(
        tf.reshape(user_gru_output, shape=[-1, gru_hidden_units[-1]]))
    q_1 = tf.reshape(q_1, shape=[-1, user_gru_output_shape[1], user_gru_output_shape[2]])

    q_2 = Dense(units=gru_hidden_units[-1], use_bias=False)(user_global_output)
    q_2 = tf.tile(tf.expand_dims(q_2, axis=1), multiples=[1, user_gru_output_shape[1], 1])

    attn_weights = Dense(1, use_bias=False)(
        tf.reshape(tf.nn.sigmoid(q_1 + q_2), shape=[-1, gru_hidden_units[-1]]))
    attn_weights = tf.reshape(attn_weights, shape=[-1, user_gru_output_shape[1]])
    attn_weights = tf.tile(tf.expand_dims(attn_weights, axis=-1), multiples=[1, 1, gru_hidden_units[-1]])

    user_local_output = tf.reduce_sum(tf.multiply(attn_weights, user_gru_output), axis=1)

    user_output = concat_func([user_global_output, user_local_output], axis=1)
    user_output = Dropout(rate=output_dropout_rate)(user_output)
    user_output = Dense(item_feature_columns[0].embedding_dim, activation=None)(user_output)

    output = SampledSoftmaxLayer(num_sampled=num_sampled)(
        [pooling_item_embedding_weight, user_output, item_features[item_feature_name]])

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", user_output)

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding",
                      get_item_embedding(pooling_item_embedding_weight, item_features[item_feature_name]))

    return model
