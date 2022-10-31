"""
Author:
    Qingliang Cai, leocaicoder@163.com
    Weichen Shen, weichenswc@163.com

Reference:
Li C, Liu Z, Wu M, et al. Multi-interest network with dynamic routing for recommendation at Tmall[C]//Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019: 2615-2623.
"""

import tensorflow as tf
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, \
    embedding_lookup, varlen_embedding_lookup, get_varlen_pooling_list, get_dense_input, build_input_features
from deepctr.layers import DNN
from deepctr.layers.utils import NoMask, combined_dnn_input
from tensorflow.python.keras.layers import Concatenate, Lambda
from tensorflow.python.keras.models import Model

from ..inputs import create_embedding_matrix
from ..layers.core import CapsuleLayer, PoolingLayer, MaskUserEmbedding, LabelAwareAttention, SampledSoftmaxLayer, \
    EmbeddingIndex
from ..utils import get_item_embedding


def shape_target(target_emb_tmp, target_emb_size):
    return tf.expand_dims(tf.reshape(target_emb_tmp, [-1, target_emb_size]), axis=-1)


def tile_user_otherfeat(user_other_feature, k_max):
    return tf.tile(tf.expand_dims(user_other_feature, -2), [1, k_max, 1])


def adaptive_interest_num(seq_len, k_max):
    try:
        log_len = tf.log1p(tf.cast(seq_len, dtype="float32"))
        log_2 = tf.log(2.)
    except AttributeError:
        log_len = tf.math.log1p(tf.cast(seq_len, dtype="float32"))
        log_2 = tf.math.log(2.)
    k_user = tf.cast(tf.maximum(
        1.,
        tf.minimum(
            tf.cast(k_max, dtype="float32"),  # k_max
            log_len / log_2  # hist_len
        )
    ), dtype="int32")
    return k_user


def MIND(user_feature_columns, item_feature_columns, k_max=2, p=100, dynamic_k=False,
         user_dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_use_bn=False, l2_reg_dnn=0, l2_reg_embedding=1e-6,
         dnn_dropout=0, output_activation='linear', sampler_config=None, seed=1024):
    """Instantiates the MIND Model architecture.

    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param k_max: int, the max size of user interest embedding
    :param p: float,the parameter for adjusting the attention distribution in LabelAwareAttention.
    :param dynamic_k: bool, whether or not use dynamic interest number
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn:  L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout:  float in [0,1), the probability we will drop out a given DNN coordinate.
    :param output_activation: Activation function to use in output layer
    :param sampler_config: negative sample config.
    :param seed: integer ,to use as random seed.
    :return: A Keras model instance.

    """

    if len(item_feature_columns) > 1:
        raise ValueError("Now MIND only support 1 item feature like item_id")
    item_feature_column = item_feature_columns[0]
    item_feature_name = item_feature_column.name
    item_vocabulary_size = item_feature_columns[0].vocabulary_size
    item_embedding_dim = item_feature_columns[0].embedding_dim
    # item_index = Input(tensor=tf.constant([list(range(item_vocabulary_size))]))

    history_feature_list = [item_feature_name]

    features = build_input_features(user_feature_columns)
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), user_feature_columns)) if user_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), user_feature_columns)) if user_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), user_feature_columns)) if user_feature_columns else []
    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)
    seq_max_len = history_feature_columns[0].maxlen
    inputs_list = list(features.values())

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed, prefix="")

    item_features = build_input_features(item_feature_columns)

    query_emb_list = embedding_lookup(embedding_matrix_dict, item_features, item_feature_columns,
                                      history_feature_list,
                                      history_feature_list, to_list=True)
    keys_emb_list = embedding_lookup(embedding_matrix_dict, features, history_feature_columns, history_fc_names,
                                     history_fc_names, to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)

    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)

    dnn_input_emb_list += sequence_embed_list

    # keys_emb = concat_func(keys_emb_list, mask=True)
    # query_emb = concat_func(query_emb_list, mask=True)

    history_emb = PoolingLayer()(NoMask()(keys_emb_list))
    target_emb = PoolingLayer()(NoMask()(query_emb_list))

    # target_emb_size = target_emb.get_shape()[-1].value
    # max_len = history_emb.get_shape()[1].value
    hist_len = features['hist_len']

    if dynamic_k:
        interest_num = Lambda(adaptive_interest_num, arguments={'k_max': k_max})(hist_len)
        high_capsule = CapsuleLayer(input_units=item_embedding_dim,
                                    out_units=item_embedding_dim, max_len=seq_max_len,
                                    k_max=k_max)((history_emb, hist_len, interest_num))
    else:
        high_capsule = CapsuleLayer(input_units=item_embedding_dim,
                                    out_units=item_embedding_dim, max_len=seq_max_len,
                                    k_max=k_max)((history_emb, hist_len))

    if len(dnn_input_emb_list) > 0 or len(dense_value_list) > 0:
        user_other_feature = combined_dnn_input(dnn_input_emb_list, dense_value_list)

        other_feature_tile = Lambda(tile_user_otherfeat, arguments={'k_max': k_max})(user_other_feature)

        user_deep_input = Concatenate()([NoMask()(other_feature_tile), high_capsule])
    else:
        user_deep_input = high_capsule

    user_embeddings = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn,
                          dnn_dropout, dnn_use_bn, output_activation=output_activation, seed=seed,
                          name="user_dnn")(
        user_deep_input)

    item_inputs_list = list(item_features.values())

    item_embedding_matrix = embedding_matrix_dict[item_feature_name]

    item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[item_feature_name])

    item_embedding_weight = NoMask()(item_embedding_matrix(item_index))

    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])

    if dynamic_k:
        user_embeddings = MaskUserEmbedding(k_max)([user_embeddings, interest_num])
        user_embedding_final = LabelAwareAttention(k_max=k_max, pow_p=p)((user_embeddings, target_emb, interest_num))
    else:
        user_embedding_final = LabelAwareAttention(k_max=k_max, pow_p=p)((user_embeddings, target_emb))

    output = SampledSoftmaxLayer(sampler_config._asdict())(
        [pooling_item_embedding_weight, user_embedding_final, item_features[item_feature_name]])
    model = Model(inputs=inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", inputs_list)
    model.__setattr__("user_embedding", user_embeddings)

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding",
                      get_item_embedding(pooling_item_embedding_weight, item_features[item_feature_name]))

    return model
