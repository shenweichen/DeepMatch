"""
Author:
    Zhe Wang,734914022@qq.com
Reference:
Huang P S , He X , Gao J , et al. Learning deep structured semantic models for web search using clickthrough data[C]// Acm International Conference on Conference on Information & Knowledge Management. ACM, 2013.
"""

import tensorflow as tf
from deepctr.feature_column import build_input_features, create_embedding_matrix
from deepctr.layers import PredictionLayer, DNN, combined_dnn_input
from deepctr.layers.utils import NoMask, combined_dnn_input
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import Model

from ..inputs import input_from_feature_columns
from ..layers.core import Similarity, EmbeddingIndex, PoolingLayer, InBatchSoftmaxLayer
from ..utils import l2_normalize, inner_product, get_item_embedding


def DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 32),
         item_dnn_hidden_units=(64, 32),
         dnn_activation='relu', dnn_use_bn=False,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, temperature=0.05,
         seed=1024, loss_type='logistic', sampler_config=None):
    """Instantiates the Deep Structured Semantic Model architecture.

    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param item_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of item tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param gamma: float. Scaling factor.
    :param seed: integer ,to use as random seed.
    :param metric: str, ``"cos"`` for  cosine  or  ``"ip"`` for inner product
    :return: A Keras model instance.

    """

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed,
                                                    seq_mask_zero=True)

    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features,
                                                                                   item_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)

    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, output_activation='linear', seed=seed)(user_dnn_input)
    user_dnn_out = l2_normalize(user_dnn_out)

    if len(item_dnn_hidden_units) > 0:
        item_dnn_out = DNN(item_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                           dnn_use_bn, output_activation='linear', seed=seed)(item_dnn_input)
    else:
        item_dnn_out = item_dnn_input
    item_dnn_out = l2_normalize(item_dnn_out)

    if loss_type == "logistic":
        score = inner_product(user_dnn_out, item_dnn_out, temperature)
        output = PredictionLayer("binary", False)(score)

    elif loss_type == "softmax":
        output = InBatchSoftmaxLayer(sampler_config._asdict(), temperature)(
            [user_dnn_out, item_dnn_out, item_features[sampler_config.item_name]])

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)

    return model
