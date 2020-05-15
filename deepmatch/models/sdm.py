# -*- coding:utf-8 -*-
"""
Author:
    Zhe Wang,734914022@qq.com

Reference:
    [1] Lv, Fuyu, Jin, Taiwei, Yu, Changlong etc. SDM: Sequential Deep Matching Model for Online Large-scale Recommender System[J].
"""

import tensorflow as tf
from collections import OrderedDict
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Lambda
from deepctr.inputs import build_input_features, SparseFeat, DenseFeat, get_varlen_pooling_list, VarLenSparseFeat, \
    create_embedding_matrix, embedding_lookup, varlen_embedding_lookup, concat_func
from deepctr.layers.utils import NoMask
from ..layers.core import PoolingLayer, SampledSoftmaxLayer, EmbeddingIndex
from ..layers.sequence import DynamicMultiRNN
from ..layers.interaction import UserAttention, SelfMultiHeadAttention, AttentionSequencePoolingLayer
from deepmatch.utils import get_item_embedding


def SDM(user_feature_columns, item_feature_columns, history_feature_list, units=64, rnn_layers=2, dropout_rate=0.2,rnn_num_res=1,
        num_head=4, l2_reg_embedding=1e-6, dnn_activation='tanh', init_std=0.0001, seed=1024, num_sampled=5):
    """Instantiates the Sequential Deep Matching Model architecture.
    """

    if len(item_feature_columns) > 1:
        raise ValueError("Now MIND only support 1 item feature like item_id")
    item_feature_column = item_feature_columns[0]
    item_feature_name = item_feature_column.name
    item_vocabulary_size = item_feature_columns[0].vocabulary_size

    features = build_input_features(user_feature_columns)

    user_inputs_list = list(features.values())

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), user_feature_columns)) if user_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), user_feature_columns)) if user_feature_columns else []
    if len(dense_feature_columns) != 0:
        raise ValueError("Now SDM don't support dense feature")
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), user_feature_columns)) if user_feature_columns else []

    sparse_varlen_feature_columns = []
    prefer_history_columns = []
    short_history_columns = []

    prefer_fc_names = list(map(lambda x: "prefer_" + x, history_feature_list))
    short_fc_names = list(map(lambda x: "short_" + x, history_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in prefer_fc_names:
            prefer_history_columns.append(fc)

        elif feature_name in short_fc_names:
            short_history_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns+item_feature_columns, l2_reg_embedding, init_std, seed,
                                                    prefix="")

    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())

    prefer_emb_list = embedding_lookup(embedding_matrix_dict, features, prefer_history_columns, prefer_fc_names,
                                      prefer_fc_names, to_list=True)  # L^u
    short_emb_list = embedding_lookup(embedding_matrix_dict, features, short_history_columns, short_fc_names,
                                      short_fc_names, to_list=True)   # S^u
    # dense_value_list = get_dense_input(features, dense_feature_columns)
    user_emb_list = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns, to_list=True)

    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)
    user_emb_list += sequence_embed_list   #e^u
    # if len(user_emb_list) > 0 or len(dense_value_list) > 0:
    #     user_emb_feature = combined_dnn_input(user_emb_list, dense_value_list)
    user_emb = concat_func(user_emb_list)
    user_emb_output = Dense(units, activation=dnn_activation, name="user_emb_output")(user_emb)


    prefer_sess_length = features['prefer_sess_length']
    prefer_att_outputs = []
    for i, prefer_emb in enumerate(prefer_emb_list):
        prefer_attention_output = AttentionSequencePoolingLayer(dropout_rate=0)([user_emb_output, prefer_emb, prefer_sess_length])
        prefer_att_outputs.append(prefer_attention_output)
    prefer_att_concat = concat_func(prefer_att_outputs)
    prefer_output = Dense(units, activation=dnn_activation, name="prefer_output")(prefer_att_concat)

    short_sess_length = features['short_sess_length']
    short_emb_concat = concat_func(short_emb_list)
    short_emb_input = Dense(units, activation=dnn_activation, name="short_emb_input")(short_emb_concat)

    short_rnn_output = DynamicMultiRNN(num_units=units, return_sequence=True, num_layers=rnn_layers, num_residual_layers=rnn_num_res,
                                       dropout_rate=dropout_rate)([short_emb_input, short_sess_length])

    short_att_output = SelfMultiHeadAttention(num_units=units, head_num=num_head, dropout_rate=dropout_rate, future_binding=True,
                                              use_layer_norm=True)([short_rnn_output, short_sess_length]) # [batch_size, time, num_units]

    short_output = UserAttention(num_units=units, activation=dnn_activation, use_res=True, dropout_rate=dropout_rate)\
        ([user_emb_output, short_att_output, short_sess_length])

    gate_input = concat_func([prefer_output, short_output, user_emb_output])
    gate = Dense(units, activation='sigmoid')(gate_input)

    gate_output = Lambda(lambda x: tf.multiply(x[0], x[1]) + tf.multiply(1-x[0], x[2]))([gate, short_output, prefer_output])
    gate_output_reshape = Lambda(lambda x: tf.squeeze(x, 1))(gate_output)

    item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[item_feature_name])
    item_embedding_matrix = embedding_matrix_dict[item_feature_name]
    item_embedding_weight = NoMask()(item_embedding_matrix(item_index))


    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])

    output = SampledSoftmaxLayer(num_sampled=num_sampled)(
        inputs=(pooling_item_embedding_weight, gate_output_reshape, item_features[item_feature_name]))
    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", gate_output)

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding", get_item_embedding(pooling_item_embedding_weight, item_features[item_feature_name]))

    return model