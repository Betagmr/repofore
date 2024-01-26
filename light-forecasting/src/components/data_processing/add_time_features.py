import pandas as pd


def add_time_features(df_raw):
    df = df_raw.copy()
    date_offset = (df.index.month * 100 + df.index.day - 320) % 1300

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["dayofmonth"] = df.index.day
    df["dayofyear"] = df.index.dayofyear
    df["season"] = pd.cut(date_offset, [0, 300, 602, 900, 1300], labels=False)

    return df
