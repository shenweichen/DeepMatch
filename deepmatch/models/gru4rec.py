import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import GRU, Dense
from tensorflow.python.keras.activations import sigmoid
import tensorflow.python.keras.backend as K
from deepctr.inputs import build_input_features, create_embedding_matrix
from deepctr.layers.core import PredictionLayer


def bpr(yTrue, yhat):
    """
    Bayesian Personalized Ranking

    """

    yhatT = K.transpose(yhat)
    return K.mean(-K.log(sigmoid(tf.linalg.diag_part(yhat) - yhatT)))


def top1(yTrue, yhat):
    """
    This is a customized loss function designed for solving the task in 'session-based recommendations
    with recurrent neural networks'

    """
    yhatT = tf.transpose(yhat)
    term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat) + yhatT) + tf.nn.sigmoid(yhatT ** 2), axis=0)
    term2 = tf.nn.sigmoid(tf.diag_part(yhat) ** 2) / len(yhat)
    return tf.reduce_mean(term1 - term2)


def GRU4REC(item_feature_columns, n_classes, gru_units, batch_size, l2_reg_embedding=1e-6, init_std=0.0001,
            seed=1024):
    """
    Instantiates the GRU for Recommendation Model architecture.

    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param n_classes: int, number of the label classes.
    :param gru_units: tuple, the layer number and units in each GRU layer.
    :param batch_size: int, number of samples in each batch.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param init_std: float, to use as the initialize std of embedding vector
    :param seed: integer, to use as random seed.
    :return: A Keras model instance.

    """

    item_feature_name = item_feature_columns[0].name

    embedding_matrix_dict = create_embedding_matrix(item_feature_columns, l2_reg_embedding,
                                                    init_std, seed, prefix="")

    item_features = build_input_features(item_feature_columns)
    item_features['movie_id'].set_shape((batch_size, 1))
    item_inputs_list = list(item_features.values())
    item_embedding_matrix = embedding_matrix_dict[item_feature_name]

    item_emb = item_embedding_matrix(item_features[item_feature_name])
    for i, j in enumerate(gru_units):
        if i == 0:
            x, gru_states = GRU(j, stateful=True, return_state=True, name='gru_{}'.format(str(i)))(item_emb)
        else:
            x, gru_states = GRU(j, stateful=True, return_state=True, name='gru_{}'.format(str(i)))(x)

        x = tf.reshape(x, (batch_size, 1, -1))

    x = tf.reshape(x, (batch_size, -1))
    x = Dense(n_classes, activation='linear')(x)

    output = PredictionLayer("multiclass", False)(x)

    model = Model(inputs=item_inputs_list, outputs=output)

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding", item_emb)

    return model
