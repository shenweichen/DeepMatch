from deepmatch.models import NCF
from ..utils import  get_xy_fd_ncf


def test_NCF():
    model_name = "NCF"

    x, y, user_feature_columns, item_feature_columns = get_xy_fd_ncf(False)
    model = NCF(user_feature_columns, item_feature_columns, )

    model.compile('adam', "binary_crossentropy")
    model.fit(x, y, batch_size=10, epochs=2, validation_split=0.5)


if __name__ == "__main__":
    pass
