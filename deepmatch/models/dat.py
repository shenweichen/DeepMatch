"""
Author:
    Yang Bo, 469828263@qq.com
Reference:
    Yantao Yu, Weipeng Wang, Zhoutian Feng, Daiyue Xue, et al. A Dual Augumented Two-tower Model for Online Large-scale Recommendation. DLP-KDD 2021.
"""

from deepctr.feature_column import build_input_features, create_embedding_matrix
from deepctr.layers import PredictionLayer, DNN, combined_dnn_input
from deepctr.layers.utils import Hash
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

from ..inputs import input_from_feature_columns
from ..layers.core import Similarity

def generate_augmented_embedding(feat, l2_reg_embedding=1e-6):
    inp = Input(shape=(1,), name='aug_inp_' + feat.name, dtype=feat.dtype)
    if feat.use_hash:
        lookup_idx = Hash(feat.vocabulary_size, mask_zero=False, vocabulary_path=feat.vocabulary_path)(inp)
    else:
        lookup_idx = inp 
    emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                                    embeddings_initializer=feat.embeddings_initializer,
                                    embeddings_regularizer=l2(l2_reg_embedding),
                                    name='aug_emb_' + feat.embedding_name)
    emb.trainable = feat.trainable
    return inp, Flatten()(emb(lookup_idx))

def DAT(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 32),
         item_dnn_hidden_units=(64, 32),
         dnn_activation='tanh', dnn_use_bn=False,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024, metric='cos'):
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
    :param seed: integer ,to use as random seed.
    :param metric: str, ``"cos"`` for  cosine  or  ``"ip"`` for inner product
    :return: A Keras model instance.

    """

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed,
                                                    seq_mask_zero=True)

    user_features = build_input_features(user_feature_columns)
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    i_u, a_u = generate_augmented_embedding(user_feature_columns[0])
    user_inputs_list = list(user_features.values()) + [i_u]
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, [a_u])

    item_features = build_input_features(item_feature_columns)
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features,
                                                                                   item_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    i_v, a_v = generate_augmented_embedding(item_feature_columns[0])
    item_inputs_list = list(item_features.values()) + [i_v]
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, [a_v])

    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed=seed)(user_dnn_input)

    item_dnn_out = DNN(item_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed=seed)(item_dnn_input)

    score = Similarity(type=metric)([user_dnn_out, item_dnn_out])

    output = PredictionLayer("binary", False)(score)

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)

    a_u_l = K.stop_gradient(a_u)
    a_v_l = K.stop_gradient(a_v)
    return model, output, user_dnn_out, item_dnn_out, a_u_l, a_v_l
