import pytest
import tensorflow as tf
from deepmatch.models import ComiRec
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
from tensorflow.python.keras import backend as K

from tests.utils import check_model, get_xy_fd


@pytest.mark.parametrize(
    'k_max,p,interest_extractor,add_pos',
    [(2, 1, 'sa', True), (1, 100, 'dr', False), (3, 50, 'dr', True),
     ]
)
def test_COMIREC(k_max, p, interest_extractor, add_pos):
    model_name = "COMIREC"

    x, y, user_feature_columns, item_feature_columns = get_xy_fd(False)

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    else:
        K.set_learning_phase(True)
    sampler_config = NegativeSampler(sampler='uniform', num_sampled=2, item_name='item')
    model = ComiRec(user_feature_columns, item_feature_columns, k_max=k_max, p=p, interest_extractor=interest_extractor,
                    add_pos=add_pos, sampler_config=sampler_config)
    model.compile('adam', sampledsoftmaxloss)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
