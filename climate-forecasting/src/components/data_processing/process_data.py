from src.components.data_processing import add_lags, add_time_features


def process_data(df):
    df_data = df.copy()
    df_data["humidity_pressure_ratio"] = df_data["humidity"] / df_data["meanpressure"]
    df_data = add_time_features(df_data, exclude=["hour"])
    df_data = add_lags(df_data)

    return df_data
