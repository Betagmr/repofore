import pandas as pd


def add_lags(df):
    new_df = df.copy()
    target_map = new_df["meantemp"].to_dict()

    new_df["lag1"] = (new_df.index - pd.Timedelta("364 days")).map(target_map)
    new_df["lag2"] = (new_df.index - pd.Timedelta("728 days")).map(target_map)
    new_df["lag3"] = (new_df.index - pd.Timedelta("1092 days")).map(target_map)

    return new_df
