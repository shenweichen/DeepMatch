"""

Author:
    Qingliang Cai,leocaicoder@163.com

Reference:
Li C, Liu Z, Wu M, et al. Multi-interest network with dynamic routing for recommendation at Tmall[C]//Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019: 2615-2623.

"""

from tensorflow.python.keras.layers import Concatenate, Flatten

from deepctr.inputs import build_input_features, create_embedding_matrix, SparseFeat, VarLenSparseFeat, DenseFeat, \
    embedding_lookup, varlen_embedding_lookup, get_varlen_pooling_list
from deepctr.layers.core import DNN
from deepctr.layers.utils import concat_func, NoMask
from deepmatch.layers.core import *


def shape_target(target_emb_tmp, target_emb_size):
    return tf.expand_dims(tf.reshape(target_emb_tmp, [-1, target_emb_size]), axis=-1)


def tile_user_otherfeat(user_other_feature, k_max):
    return tf.tile(tf.expand_dims(user_other_feature, -2), [1, k_max, 1])


def MIND(dnn_feature_columns, history_feature_list, target_song_size, k_max=2, dnn_use_bn=False,
         user_hidden_unit=64, dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0,
         init_std=0.0001, seed=1024):
    """
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param target_song_size: int, the total size of the recall songs
    :param k_max: int, the max size of user interest embedding
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param user_hidden_unit: int. user dnn hidden layer size
    :param dnn_activation: Activation function to use in deep net
    :param l2_reg_dnn:  L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout:  float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :return:
    """
    features = build_input_features(dnn_feature_columns)
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    hist_len = features['hist_len']

    inputs_list = list(features.values())
    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, init_std, seed, prefix="")
    history_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, history_fc_names,
                                        history_fc_names, to_list=True)
    history_emb = concat_func(history_emb_list, mask=False)

    target_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, ['item'],
                                       history_feature_list, to_list=True)
    target_emb_tmp = concat_func(target_emb_list, mask=False)
    target_emb_size = target_emb_tmp.get_shape()[-1].value

    target_emb = tf.keras.layers.Lambda(shape_target, arguments={'target_emb_size': target_emb_size})(target_emb_tmp)

    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)
    dnn_input_emb_list += sequence_embed_list

    deep_input_emb = concat_func(dnn_input_emb_list)
    user_other_feature = Flatten()(deep_input_emb)

    max_len = history_emb.get_shape()[1].value

    high_capsule = CapsuleLayer(input_units=target_emb_size,
                                out_units=target_emb_size, max_len=max_len,
                                k_max=k_max)((history_emb, hist_len))
    other_feature_tile = tf.keras.layers.Lambda(tile_user_otherfeat, arguments={'k_max': k_max})(user_other_feature)

    user_deep_input = Concatenate()([NoMask()(other_feature_tile), high_capsule])

    user_embeddings = DNN((user_hidden_unit, target_emb_size), dnn_activation, l2_reg_dnn,
                          dnn_dropout, dnn_use_bn, seed, name="user_embedding")(user_deep_input)

    k_user = tf.cast(tf.maximum(
        1.,
        tf.minimum(
            tf.cast(k_max, dtype="float32"),
            tf.log1p(tf.cast(hist_len, dtype="float32")) / tf.log(2.)
        )
    ), dtype="int64")  # [B,1] forword/Cast_2

    user_embedding_final = DotProductAttentionLayer(shape=[target_emb_size, target_emb_size])(
        (user_embeddings, target_emb), seq_length=k_user, max_len=k_max
    )

    output = SampledSoftmaxLayer(target_song_size=target_song_size, target_emb_size=target_emb_size)(
        inputs=(user_embedding_final, features['item']))

    model = Model(inputs=inputs_list, outputs=output)
    return model
