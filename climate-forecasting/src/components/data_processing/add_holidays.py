from pandas.tseries.holiday import USFederalHolidayCalendar as USCalendar


def add_holidays(df_raw, calendar=USCalendar()):
    df = df_raw.copy()
    holidays = calendar.holidays(
        start=df.index.min(),
        end=df.index.max(),
    )
    df["holiday"] = df.index.isin(holidays)

    return df
