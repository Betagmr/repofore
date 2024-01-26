from xgboost import XGBRegressor

from src.settings.params import XGBOOST_PARAMS


def train_XGBoost(x_train, y_train, x_test, y_test) -> XGBRegressor:
    model = XGBRegressor(**XGBOOST_PARAMS)

    return model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=True,
    )
