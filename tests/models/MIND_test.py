import tensorflow as tf
from deepmatch.models import MIND
from deepmatch.utils import sampledsoftmaxloss
from tensorflow.python.keras import backend as K

from ..utils import check_model, get_xy_fd


@pytest.mark.parametrize(
    'dynamic_k,p',
    [(False, 1), (True, 100)
     ]
)
def test_MIND(dynamic_k, p):
    model_name = "MIND"

    x, y, user_feature_columns, item_feature_columns = get_xy_fd(False)
    K.set_learning_phase(True)

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    model = MIND(user_feature_columns, item_feature_columns, num_sampled=2, p=p, dynamic_k=dynamic_k,
                 user_dnn_hidden_units=(16, 4))

    model.compile('adam', sampledsoftmaxloss)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
