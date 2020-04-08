import numpy as np

from deepmatch.models import DSSM
from deepmatch.utils import sampledsoftmaxloss
from deepctr.inputs import SparseFeat,VarLenSparseFeat,DenseFeat,get_feature_names
from tensorflow.python.keras import backend as K
from ..utils import check_model,get_xy_fd


#@pytest.mark.xfail(reason="There is a bug when save model use Dice")
#@pytest.mark.skip(reason="misunderstood the API")


def test_DSSM():
    model_name = "DSSM"

    x, y, user_feature_columns, item_feature_columns = get_xy_fd(False)
    model = DSSM(user_feature_columns, item_feature_columns, )

    model.compile('adam', "binary_crossentropy")
    check_model(model,model_name,x,y,check_model_io=False)


if __name__ == "__main__":
    pass
