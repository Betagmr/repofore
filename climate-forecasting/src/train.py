import pandas as pd
from clearml import Task
from matplotlib import pyplot as plt

from src.components.data_processing import process_data
from src.components.metrics import mean_absolute_percentage_error
from src.components.model.train_model import train_XGBoost
from src.components.visualization import compare_prediction, plot_feature_importance
from src.settings.metadata import TEST_PATH, TRAIN_PATH
from src.settings.params import XGBOOST_PARAMS


def read_data(train_path, test_path):
    df_train = pd.read_csv(TRAIN_PATH, index_col=[0], parse_dates=[0])
    df_test = pd.read_csv(TEST_PATH, index_col=[0], parse_dates=[0])

    df_train.iloc[-1, 0:3] = df_test.iloc[0, 0:3]
    df_test = df_test.iloc[1:, :]

    return pd.concat([df_train, df_test])


def train(task: Task):
    # Load data
    df_data = read_data(TRAIN_PATH, TEST_PATH)

    # Process data
    df_data = process_data(df_data)

    # Split data
    split_date = "2017-01-01"
    df_train = df_data.loc[df_data.index < split_date].copy()
    df_test = df_data.loc[df_data.index >= split_date].copy()

    # Train model
    target, *features = df_train.columns
    x_train, y_train = df_train[features], df_train[target]
    x_test, y_test = df_test[features], df_test[target]

    model = train_XGBoost(x_train, y_train, x_test, y_test)
    model.save_model("model.json")

    # Visualize predictions
    y_test_predictions = model.predict(x_test)
    df_visual = df_data.loc[:, ["meantemp"]]
    df_visual.loc[df_visual.index >= split_date, "prediction"] = y_test_predictions

    # Report metrics and visualizations
    mape = mean_absolute_percentage_error(y_test, y_test_predictions)
    task.logger.report_scalar(title="MAPE", series="MAPE", iteration=0, value=mape)

    plot_feature_importance(model.feature_importances_, model.feature_names_in_)
    plt.show()

    compare_prediction(df_visual, "date", df_visual.index[0], df_visual.index[-1])
    plt.show()

    # Log into debug samples
    fig_list = [
        compare_prediction(df_visual, "date", "2017-01-01", "2017-01-07")[0],
        compare_prediction(df_visual, "date", "2017-02-01", "2017-04-07")[0],
        compare_prediction(df_visual, "date", "2017-03-01", "2017-05-07")[0],
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
        project_name="Weather Forecasting",
        task_name="Train",
    )

    task.connect(XGBOOST_PARAMS, "XGBoost parameters")

    train(task)
