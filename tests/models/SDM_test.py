from deepmatch.models import SDM
from deepmatch.utils import sampledsoftmaxloss
from tensorflow.python.keras import backend as K
from tests.utils import check_model, get_xy_fd_sdm
import tensorflow as tf


#@pytest.mark.xfail(reason="There is a bug when save model use Dice")
#@pytest.mark.skip(reason="misunderstood the API")


def test_SDM():
    model_name = "SDM"
    tf.keras.backend.set_learning_phase(1)
    x, y, user_feature_columns, item_feature_columns, history_feature_list = get_xy_fd_sdm(False)
    K.set_learning_phase(True)

    if tf.__version__ >= '2.0.0':
       tf.compat.v1.disable_eager_execution()

    model = SDM(user_feature_columns, item_feature_columns, history_feature_list, units=64)
    model.summary()

    model.compile('adam', sampledsoftmaxloss)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    test_SDM()
