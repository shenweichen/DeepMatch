import pytest
import tensorflow as tf
from deepmatch.models import YoutubeDNN
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
from tensorflow.python.keras import backend as K
from tests.utils import check_model, get_xy_fd


@pytest.mark.parametrize(
    'sampler',
    ['inbatch', 'uniform', 'frequency', 'adaptive',
     ]
)
def test_YoutubeDNN(sampler):
    model_name = "YoutubeDNN"

    x, y, user_feature_columns, item_feature_columns = get_xy_fd(False)

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    else:
        K.set_learning_phase(True)
    from collections import Counter
    train_counter = Counter(x['item'])
    item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
    sampler_config = NegativeSampler(sampler, num_sampled=2, item_name='item', item_count=item_count, distortion=1.0)
    model = YoutubeDNN(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(16, 4),
                       sampler_config=sampler_config)
    model.compile('adam', sampledsoftmaxloss)

    check_model(model, model_name, x, y, check_model_io=True)


if __name__ == "__main__":
    pass
