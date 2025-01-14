import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import streamlit as st
from data_processing import utils as us


class CalculateData:
    def __init__(self, df):
        self.df = df.copy()


    def step_average(self, x):
        """x日間隔で平均値を計算して新しいdfを作成"""
        # ステップごとの値を抽出
        temp_array = self.df.values[::x]
        adjudted_index = self.df.index[::x]
        average = temp_array.mean(axis=-1)
        df_average = pd.DataFrame(temp_array, index=adjudted_index, columns=self.df.columns)

        return df_average


    def calc_MA(self, x):
        """x日移動平均を計算して，新しいDFを作成"""
        #df_ma = self.df.rolling(window=x).mean().dropna(how='any')
        temp_array = self.df.values
        temp_ma = sliding_window_view(temp_array, x, axis=0).mean(axis=-1)
        adjusted_index = self.df.index[x-1:]
        df_ma = pd.DataFrame(temp_ma, index=adjusted_index, columns=self.df.columns)
        #assert np.allclose(temp_ma, df_ma.values), "NumPyとPandasの結果が一致しません！"

        return df_ma


    def temp_curve(self):
        """年平均地表面温度から温度変化曲線を計算（松岡, 1991）"""
        # 変数の定義
        Tm = self.df.resample("Y").mean()   # 年平均気温
        Tmax = self.df.resample("Y").max()  # 年最高気温
        Tmin = self.df.resample("Y").min()  # 年最低気温
        To = (Tmax - Tmin) / 2    # 気温の年振幅
        t = np.arange(0, 366)   # タイムスケール (0~365日)

        # cos(x)の計算（年周期のため、365日分）
        x = (2 * np.pi * t) / 365  # 1年365日周期のラジアン変換
        cos_x = np.cos(x)[None, :]  # 365日分のcos(x)を計算, １次元の横ベクトル

        # Tm, Toをnumpy配列に変換 (年数×深度数の行列)
        Tm_values = Tm.values.T  # shape: (深度数, 年数)
        To_values = To.values.T  # shape: (深度数, 年数)
        #st.write(Tm)
        #st.write(Tm_values)

        # 温度変化曲線の計算
        temp_curve = {}
        for i in range(Tm_values.shape[1]):  # 各年ごとのループ
            # 各年のTmとToに対して計算
            T = Tm_values[:, i][:, None] + To_values[:, i][:, None] * cos_x
            #st.write(cos_x.shape)
            #st.write(T.shape)
            T_df = pd.DataFrame(T.T, columns=Tm.columns)
            #T_df["year"] = Tm.index[i].year

            # 結果を辞書に保存（年ごとのラベル付き）
            temp_curve[f"{Tm.index[i].year}"] = T_df  # transposeして返す

        #st.write(temp_curve)
        return temp_curve


    def calc_FI(self, n=0.0):
        """FI or AFIを計算
        Args:
            FI: t < 0 となるtをひたすら積算
            AFI: "初めて" t < 0 となった日からFIを計算（デフォはn=0）
        """
        df = self.df
        result = pd.DataFrame(index=df.index)     # 結果を保存するdf
        columns_name = self.df.columns  # カラム名を取得

        # FIの計算
        for col in columns_name:
            result[f"{col}_FI"] = abs(df[col].where(df[col]<=n, n)).cumsum()

        # 欠損値の処理
        result.iloc[0] = result.iloc[0].fillna(0)
        result = result.ffill()

        return result


    def calc_TDD(self):
        """TDDの計算"""
        df = self.df.copy()
        result = pd.DataFrame(index=df.index)     # 結果を保存するdf
        # TDDの計算
        columns_name = self.df.columns  # list
        for col in columns_name:
            result[f"{col}_TDD"] = df[col].mask(df[col]<=0, 0).cumsum()

        return result


    def get_group(self, freq_key="M"):
        """
        入力dfはDatetimeIndex指定．
        頻度ごとにDataFrameを辞書形式で返す．

        Args:
            freq = {
            "D": "day",
            "W": "week",
            "M": "month",
            "Q": "quarter",
            "Y": "year"
            }
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

        # 集計頻度を表す列を追加
        df = self.df.copy()
        df[f"{freq[freq_key]}"] = df.index.to_period



# 条件式をdfに適用して抽出するクラス
class ApplyCondition:
    def __init__(self, df, freq_key="D"):
        """_summary_ 

        Args:
            df (_type_): _description_
            freq_key (str, optional): _description_. Defaults to "D".
        """

        self.freq_key = freq_key
        self.df = df.copy()
        self.df_mean = self.df.copy().resample(freq_key).mean()
        self.df_max = self.df.copy().resample(freq_key).max()
        self.df_min = self.df.copy().resample(freq_key).min()
        self.processed_df = None


    def judge_freeze(self):
        """FT判定したdfを出力"""
        new_df = pd.DataFrame(index=self.df_mean.index)
        columns_name = self.df_mean.columns

        # FT判定
        for col in columns_name:
            new_df[f"{col}_FT1"] = ((self.df_min[f"{col}"] < 0)) & (self.df_max[f"{col}"] >= 0).astype(int)   # x<0, 0<=x
            new_df[f"{col}_FT2"] = ((self.df_min[f"{col}"] < -2)) & (self.df_max[f"{col}"] > 2).astype(int)   # x<-2, 2<x
            new_df[f"{col}_FT3"] = ((self.df_min[f"{col}"] >= -8)) & (self.df_max[f"{col}"] < -3).astype(int)  # -8<= x <-3
            # 杉山さんのリプ待ち
            #new_df[f"{col}_FT4"] = ((self.df_min[F"{col}"] ))

        # Nanを0に置換
        new_df.fillna(0, inplace=True)
        self.processed_df = new_df

        return self.processed_df


    def get_count(self):
        """カウント列を新しいdfとして出力"""
        if self.processed_df is None:
            raise ValueError("先にデータ処理メソッドを呼び出して下さい")
        df = self.processed_df.copy()
        df.loc["count"] = df.sum()
        count = df.loc["count"]
        return count




