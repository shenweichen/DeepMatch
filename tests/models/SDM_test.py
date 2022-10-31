import tensorflow as tf
from deepmatch.models import SDM
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
from tensorflow.python.keras import backend as K

from ..utils import check_model, get_xy_fd_sdm


def test_SDM():
    model_name = "SDM"
    x, y, user_feature_columns, item_feature_columns, history_feature_list = get_xy_fd_sdm(False)

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
        #tf.compat.v1.disable_v2_behavior()
    else:
        K.set_learning_phase(True)

    sampler_config = NegativeSampler(sampler='uniform', num_sampled=2, item_name='item')
    model = SDM(user_feature_columns, item_feature_columns, history_feature_list, units=8,
                sampler_config=sampler_config)
    # model.summary()

    model.compile('adam', sampledsoftmaxloss)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
