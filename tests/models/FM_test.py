from deepmatch.models import FM
from ..utils import check_model, get_xy_fd


def test_FM():
    model_name = "FM"

    x, y, user_feature_columns, item_feature_columns = get_xy_fd(False)
    model = FM(user_feature_columns, item_feature_columns, )

    model.compile('adam', "binary_crossentropy")
    check_model(model, model_name, x, y,)


if __name__ == "__main__":
    pass
