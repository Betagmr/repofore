import pandas as pd
from matplotlib import pyplot as plt


def plot_feature_importance(feature_importance, labels):
    fig, ax = plt.subplots(figsize=(10, 5))

    df_feature_importance = pd.DataFrame(
        data=feature_importance,
        index=labels,
        columns=["importance"],
    ).sort_values(by="importance")

    df_feature_importance.plot.barh(ax=ax)
    ax.set_title("Feature importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    return fig, ax
