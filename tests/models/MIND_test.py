import numpy as np

from deepmatch.models import MIND
from deepmatch.utils import sampledsoftmaxloss
from deepctr.inputs import SparseFeat,VarLenSparseFeat,DenseFeat,get_feature_names
from tensorflow.python.keras import backend as K
from ..utils import check_model,get_xy_fd


#@pytest.mark.xfail(reason="There is a bug when save model use Dice")
#@pytest.mark.skip(reason="misunderstood the API")


# def test_MIND():
#     model_name = "MIND"
#
#     x, y, user_feature_columns, item_feature_columns = get_xy_fd(False)
#     K.set_learning_phase(True)
#     model = MIND(user_feature_columns, item_feature_columns, num_sampled=2, user_dnn_hidden_units=(16, 4))
#
#     model.compile('adam', sampledsoftmaxloss)
#     check_model(model,model_name,x,y,check_model_io=True)
#

if __name__ == "__main__":
    pass
