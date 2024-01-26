from matplotlib import pyplot as plt


def compare_prediction(df, date_column, date_start, date_end):
    fig, ax = plt.subplots(figsize=(10, 5))

    df_query = df.query(
        f"{date_column} >= '{date_start}' and {date_column} < '{date_end}'"
    )
    df_query.plot(ax=ax)
    ax.set_title("Compare prediction")

    return fig, ax
