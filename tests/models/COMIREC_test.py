import pytest
import tensorflow as tf
from deepmatch.models import ComiRec
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
from tensorflow.python.keras import backend as K

from ..utils import check_model, get_xy_fd


@pytest.mark.parametrize(
    'interest_num,p',
    [(False, 1), (True, 100)
     ]
)
def test_COMIREC(interest_num, p):
    model_name = "COMIREC"

    x, y, user_feature_columns, item_feature_columns = get_xy_fd(False)

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    else:
        K.set_learning_phase(True)
    sampler_config = NegativeSampler(sampler='uniform', num_sampled=2, item_name='item')
    model = ComiRec(user_feature_columns, item_feature_columns, p=p, interest_num=interest_num, 
                 user_dnn_hidden_units=(64, 32, 4), sampler_config=sampler_config)

    model.compile('adam', sampledsoftmaxloss)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
