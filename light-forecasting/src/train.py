import pandas as pd
from clearml import Task
from matplotlib import pyplot as plt

from src.components.data_processing import add_time_features
from src.components.metrics import mean_absolute_percentage_error
from src.components.model.train_model import train_XGBoost
from src.components.visualization import compare_prediction, plot_feature_importance
from src.settings.metadata import DATASET_PATH
from src.settings.params import XGBOOST_PARAMS


def train(task: Task):
    # Load data
    df_raw = pd.read_csv(DATASET_PATH, index_col=[0], parse_dates=[0])

    # Process data
    split_date = "2015-01-01"
    df_train = df_raw.loc[df_raw.index < split_date].copy()
    df_test = df_raw.loc[df_raw.index >= split_date].copy()

    train_data = add_time_features(df_train)
    test_data = add_time_features(df_test)

    # Train model
    target, *features = train_data.columns
    x_train, y_train = train_data[features], train_data[target]
    x_test, y_test = test_data[features], test_data[target]

    model = train_XGBoost(x_train, y_train, x_test, y_test)
    model.save_model("model.json")

    df_test["prediction"] = model.predict(x_test)
    df_visual = df_raw.merge(
        df_test["prediction"],
        how="left",
        left_index=True,
        right_index=True,
    )

    # Report metrics and visualizations
    mape = mean_absolute_percentage_error(y_test, df_test["prediction"])
    task.logger.report_scalar(title="MAPE", series="MAPE", iteration=0, value=mape)

    plot_feature_importance(model.feature_importances_, model.feature_names_in_)
    compare_prediction(df_visual, "Datetime", df_visual.index[0], df_visual.index[-1])
    plt.show()

    # Log into debug samples

    fig_list = [
        compare_prediction(df_visual, "Datetime", "2015-01-01", "2015-01-07")[0],
        compare_prediction(df_visual, "Datetime", "2016-01-01", "2016-01-07")[0],
        compare_prediction(df_visual, "Datetime", "2017-01-01", "2017-01-07")[0],
    ]

    for i, fig in enumerate(fig_list):
        task.logger.report_matplotlib_figure(
            figure=fig,
            iteration=0,
            title=fig.axes[0].get_title() + " " + str(i),
            series="debug samples",
        )


if __name__ == "__main__":
    task = Task.init(
        project_name="Time Series Forecasting",
        task_name="Train",
    )

    task.connect(XGBOOST_PARAMS, "XGBoost parameters")

    train(task)
