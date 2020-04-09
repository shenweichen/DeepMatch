"""
Author:
    Weichen Shen, wcshen1994@163.com
Reference:
Covington P, Adams J, Sargin E. Deep neural networks for youtube recommendations[C]//Proceedings of the 10th ACM conference on recommender systems. 2016: 191-198.
"""
import tensorflow as tf
from deepctr.inputs import input_from_feature_columns, build_input_features, combined_dnn_input, create_embedding_matrix
from deepctr.layers.core import DNN
from deepctr.layers.utils import NoMask
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Lambda

from deepmatch.utils import get_item_embedding, get_item_embeddingv2
from deepmatch.layers import PoolingLayer
from ..inputs import input_from_feature_columns
from ..layers.core import SampledSoftmaxLayer, SampledSoftmaxLayerv2


def YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5,
               user_dnn_hidden_units=(64, 16),
               dnn_activation='relu', dnn_use_bn=False,
               l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, ):
    """Instantiates the YoutubeDNN Model architecture.

    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param num_sampled: int, the number of classes to randomly sample per batch.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :return: A Keras model instance.

    """

    if len(item_feature_columns) > 1:
        raise ValueError("Now YoutubeNN only support 1 item feature like item_id")
    item_feature_name = item_feature_columns[0].name
    item_vocabulary_size = item_feature_columns[0].vocabulary_size
    item_index = Input(tensor=tf.constant([list(range(item_vocabulary_size))]))
    # item_idskey = Input( tensor=tf.constant([[0]*209]), name="item_idx")

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    init_std, seed, prefix="")

    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding, init_std, seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed, )(user_dnn_input)

    # item_embedding = NoMask()(embedding_matrix_dict[item_feature_name](item_idx))
    item_embedding_matrix = embedding_matrix_dict[
        item_feature_name]  # Lambda( lambda x:embedding_matrix_dict[item_feature_name])(None)
    # if not item_embedding.built:
    #     item_embedding.build()
    item_embedding_weight = item_embedding_matrix(item_index)
    # item_embedding_skey = item_embedding(item_idskey)

    # item_embedding_ip = Input(tensor=item_embedding.embeddings,name=item_feature_name+"_embeddings")
    item_embedding_weight = Lambda(lambda x: tf.squeeze(x, axis=0))(NoMask()(item_embedding_weight))
    # item_embedding_skey = Lambda(lambda x: tf.squeeze(x, axis=0))(NoMask()(item_embedding_skey))

    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])

    # print(item_idx,"itemidxitem_idxitem_idxitem_idxitem_idx")
    # print(item_embedding_ip, "item_embedding_ipitem_embedding_ipitem_embedding_ipitem_embedding_ip")
    # item_embedding = get_item_embeddingv2()

    # if not item_embedding.built:
    #     item_embedding.build([])

    output = SampledSoftmaxLayerv2(num_sampled=num_sampled)(
        inputs=(pooling_item_embedding_weight, user_dnn_out, item_features[item_feature_name]))
    model = Model(inputs=user_inputs_list + item_inputs_list + [item_index], outputs=output)  # +[item_idx]

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding", get_item_embeddingv2(item_embedding_weight, item_features[item_feature_name]))

    return model
