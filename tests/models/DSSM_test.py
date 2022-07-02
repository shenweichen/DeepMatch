import pytest
from deepmatch.models import DSSM
from deepmatch.utils import sampledsoftmaxloss,Sampler
from ..utils import check_model, get_xy_fd


@pytest.mark.parametrize(
    'loss_type',
    ['logistic','softmax'
     ]
)
def test_DSSM(loss_type):
    model_name = "DSSM"

    x, y, user_feature_columns, item_feature_columns = get_xy_fd(False)
    if loss_type == "logistic":
        model = DSSM(user_feature_columns, item_feature_columns, loss_type=loss_type)
        model.compile('adam', "binary_crossentropy")
    else:
        from collections import Counter
        item_name = 'item'
        train_counter = Counter(x[item_name])
        item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
        sampler_config = Sampler(sampler='batch', num_sampled=2, item_name=item_name,item_count=item_count, distortion=1.0)
        model = DSSM(user_feature_columns, item_feature_columns, loss_type=loss_type,sampler_config=sampler_config)
        model.compile('adam',sampledsoftmaxloss)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
