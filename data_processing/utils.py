import pandas as pd





def add_count_row(df):
    """dfの最終行に合計値行を追加"""
    df_count = df.copy()
    df_count.loc["count"] = df.sum(numeric_only=True)

    return df_count


def df_groups(df, freq_key="M"):
    """
    入力dfはDatetimeIndex指定．
    頻度ごとにDataFrameを辞書形式で返し、各DataFrameに 'count' 行を追加．

    Args:
        freq_key (str): 'D', 'W', 'M', 'Q', 'Y' のいずれかで頻度を指定
    Returns:
        dict: {期間: DataFrame} の辞書
    """
    # freq_keyと頻度の対応関係
    freq = {
        "D": "day",
        "W": "week",
        "M": "month",
        "Q": "quarter",
        "Y": "year"
    }

    if freq_key not in freq:
        raise ValueError(f"無効な頻度キー: {freq_key}")

    df = df.copy()
    # datetimeインデックスでない場合は変換
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df[f"{freq[freq_key]}"] = df.index.to_period(freq_key)

    # 頻度を表す列を追加
    df[f"{freq[freq_key]}"] = df.index.to_period(freq_key)

    # グループ化して 'count' 行を追加
    grouped = df.groupby(f"{freq[freq_key]}")
    dfs = {
        str(period): add_count_row(group.drop(columns=[f"{freq[freq_key]}"]))
        for period, group in grouped
    }

    return dfs
