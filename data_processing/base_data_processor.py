import pandas as pd
import numpy as np
from scipy import stats
import altair as alt
import streamlit as st


# 共通のデータ処理を行うクラス
class BaseDataProcessor:
    """データ処理の基底クラス"""
    def __init__(self, file=None):
        self.file = file
        self.df = None


    def load_data(self):
        """ロード処理は各サブクラスで実装"""
        raise NotImplementedError("load_dataはサブクラスで実装してください")


    def resample_data(self, column_settings):
        """データのリサンプリング処理"""
        try:
            self.resampled_dfs = {}
            for column, settings in column_settings.items():
                # st.write(f"リサンプリング対象カラム： {column}")
                freq = settings["freq"]
                method = settings["method"]

                # リサンプル処理の実行
                if column in self.df.columns:
                    # st.write("現在のデータフレーム:")
                    # st.write(self.df.head())
                    # st.write("利用可能なカラム:")
                    # st.write(self.df.columns.tolist())

                    # st.write(f"カラム '{column}' が存在します．リサンプリングを開始します")
                    # インデックスがdatetimeか確認
                    if not pd.api.types.is_datetime64_any_dtype(self.df.index):
                        st.error("インデックスがdatetime型ではありません")
                        st.write(self.df.index)
                        raise ValueError("インデックスがdatetime型ではありません")
                    # columnのデータ型の確認
                    # st.write(f"{column}のデータ型：{self.df[column].dtype}")
                    # リサンプル処理の実行
                    try:
                        tmp = self.df[column].resample(freq).agg(method)
                        tmp = tmp.to_frame()      # Seriesをdfに変換
                        # リサンプルの結果をdictに保存（tmpはDataFrame）
                        result_key = f"{column}_{freq}_{method}"
                        # st.write(f"リサンプル後のデータ型：{type(tmp)}")
                        self.resampled_dfs[result_key] = tmp

                        # st.write(f"{column}のリサンプリング結果")
                        # st.write(tmp.head())
                    except Exception as e:
                        st.error(f"{column}のリサンプリング中にエラーが発生しました：{e}")
                else:
                    st.warning(f"カラム '{column}' は存在しません，スキップします")

            return self.resampled_dfs
        except Exception as e:
            st.error(f"リサンプリング中にエラーが発生しました：{e}")
            raise


# AWSデータの処理を行うサブクラス
class AWSDataProcessor(BaseDataProcessor):
    def __init__(self, file):
        super().__init__(file)  #親クラスのコンストラクタを呼び出す

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file, skiprows=[0, 1, 2, 4, 5], engine="python", encoding="shift-jis")
            # st.write(self.df.head())

            # 不要列の削除
            columns_drop = ["Unnamed: 1", "Unnamed: 13", "Unnamed: 14",
                            "Unnamed: 15","Unnamed: 16", "Unnamed: 17",
                            "Unnamed: 18", "Unnamed: 19", "Unnamed: 20",
                            "Unnamed: 21", "本体電力"]
            self.df = self.df.drop(columns=columns_drop, errors='ignore')

            # 列名の変更
            self.df.rename(columns={
                "Unnamed: 0": "datetime",
                "風速_10分平均": "ws(m/s)",
                "風向き": "wd(degree)",
                "最大風速": "ws_max(m/s)",
                "気温": "temp(℃)",
                "湿度": "hum(％)",
                "大気圧": "pl(hPa)",
                "雨量": "prec(mm)",
                "下向き短波放射": "dwir(W/m2)",
                "上向き短波放射": "upir(W/m2)",
                "下向き長波放射": "dwlr(W/m2)",
                "上向き長波放射": "uplr(W/m2)"
            }, inplace=True)

            # 欠損値の処理と日付インデックスに指定
            self.df["datetime"] = pd.to_datetime(self.df["datetime"], errors='coerce')
            self.df.dropna(subset=["datetime"], inplace=True)   # datatimeがNanの行を削除
            self.df.set_index("datetime", inplace=True)
            self.df = self.df.apply(pd.to_numeric, errors='coerce').dropna()

            # st.write("初期処理が完了しました：")
            # st.write(self.df)
            return self.df

        except Exception as e:
            print(f"AWSデータの処理に失敗しました： {e}")


# 気象庁のデータの処理を扱うクラス
class AMDDataProcessor(BaseDataProcessor):
    def __init__(self, file=None):
        super().__init__(file)  # 親クラスのコンストラクタを呼び出す

    def load_data(self):
        try:
            self.df = pd.read_csv(
                self.file,
                skiprows=[0, 1, 2, 4, 5],
                engine="python"
            )
            st.write(self.df)
            pass        # 後で実装
        except Exception as e:
            st.write(f"AMeDASデータの処理に失敗しました：  {e}")


# おんどとりのデータの処理を行うサブクラス
class ODTDataProcessor(BaseDataProcessor):
    def __init__(self, file):
        super().__init__(file)  #親クラスのコンストラクタを呼び出す

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file, skiprows=[1, 2], engine="python", encoding="shift-jis")
            # st.write(self.df)
            # 不要列の削除
            self.df = self.df.drop(columns=self.df.columns[[1]], errors='ignore')
            # 列名の変更
            self.df.rename(columns={
                "Date/Time": "datetime",
                "No.1": "temp(℃)",
            }, inplace=True)

            # 日付インデックスに指定
            self.df["datetime"] = pd.to_datetime(self.df["datetime"], errors='coerce')
            self.df.dropna(subset=["datetime"], inplace=True)
            self.df.set_index("datetime", inplace=True)
            self.df = self.df.apply(pd.to_numeric, errors='coerce').dropna()

            # st.write("初期処理が完了しました：")
            # st.write(self.df)
            return self.df

        except Exception as e:
            print(f"おんどとりデータの処理に失敗しました： {e}")



# グラフの描画を管理するクラス
class DataVisualizer:
    def __init__(self):
        pass

    # 散布図レイヤー
    def scatter_plot(self, df, x_column, y_column, x_range=None, y_range=None):
        """散布図の描画"""
        try:
            # 相関係数の計算
            corr = np.corrcoef(df[x_column], df[y_column])[0, 1]
            corr_val = f"Correlation: {corr:.2f}"

            # 軸範囲の設定
            x_scale = alt.Scale(domain=x_range) if x_range is not None else alt.Undefined
            y_scale = alt.Scale(domain=y_range) if y_range is not None else alt.Undefined

            # 散布図の作成
            scatter_plot = alt.Chart(df).mark_circle().encode(
                x=alt.X(f"{x_column}:Q", scale=x_scale),
                y=alt.Y(f"{y_column}:Q", scale=y_scale),
                tooltip=["datetime:T", f"{x_column}:Q", f"{y_column}:Q"],
                color=alt.Color(
                    f"{x_column}:Q",
                    scale=alt.Scale(scheme="viridis"),
                    legend=alt.Legend(title="Value")
                )
            ).properties(
                width=600, height=500
            )
            # 線形回帰分析
            slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_column], df[y_column])  # 単回帰分析
            if intercept >= 0:
                reg_eq = f"y = {slope:.2f}x + {intercept:.2f}"   # 回帰直線
            else:
                reg_eq = f"y = {slope:.2f}x {intercept:.2f}"    # 切片が負のとき
            # 回帰直線の方程式をテキストとして挿入
            reg_text = alt.Chart(pd.DataFrame({"text": [reg_eq]})).mark_text(
                align="left",
                baseline="bottom",
                fontSize=20,
                fontStyle='italic',
                color='gray'
            ).encode(
                text="text:N",
                x=alt.value(13),
                y=alt.value(50)  # 方程式の位置調整
            )
            # 回帰直線の描画
            pr_line = (
                scatter_plot.transform_regression(
                    on=f"{y_column}",
                    regression=f"{x_column}",
                    method="linear"
                    ).mark_line(size=3, opacity=0.6).encode(
                        strokeDash=alt.value([9, 9]),
                        color=alt.value("deeppink")
                    )
            )
            # 散布図と線形回帰を重ねる
            scatter_plot = scatter_plot + pr_line + reg_text
            # 相関係数を表示
            corr_text = alt.Chart(pd.DataFrame({"text": [corr_val]})).mark_text(
                align="left",
                baseline="top",
                fontSize=20,
                fontStyle='italic',
                color='gray'
            ).encode(
                text="text:N",
                x=alt.value(10),
                y=alt.value(10)
            )
            # 散布図とテキストを重ねる
            chart = scatter_plot + corr_text
            return chart.interactive()
        except Exception as e:
            st.error(f"散布図レイヤーの描画中にエラーが発生しました：{e}")

    # 箱ひげ図レイヤー
    def box_plot(self, df):
        """箱ひげ図の描画"""
        try:
            if len(df["Category"].unique()) == 1:
                chart = alt.Chart(df).mark_boxplot().encode(
                    x=alt.X("Category:N", title=None),
                    y=alt.Y("Value:Q", title="Value")
                )
            else:
                chart = alt.Chart(df).mark_boxplot().encode(
                    x=alt.X("Category:N", title=None),
                    y=alt.Y("Value:Q", title="Value"),
                    color=alt.Color(
                        "Category:N",
                        scale=alt.Scale(scheme="tableau10"),
                        legend=alt.Legend(title="Category")
                        )
                ).properties(width=600, height=520)
            return chart
        except Exception as e:
            st.error(f"箱ひげ図レイヤーの描画中にエラーが発生しました：{e}")


    # ヒストグラムレイヤー
    def hist_plot(self, df, bins):
        """ヒストグラムの描画"""
        try:
            # ビンの幅を計算
            chart_width = 600
            def sigmoid(x):
                y = 1 / (1 + np.e**-x)
                return y
            bin_width = (chart_width / bins*0.7*sigmoid(1.702*bins))
            chart = alt.Chart(df).mark_bar(size=bin_width).encode(
                x=alt.X('bin_start:Q', title='Value range'),
                y=alt.Y('count:Q', title='Frequency')
            ).properties(width=chart_width, height=400)
            return chart

        except Exception as e:
            st.error(f"ヒストグラムレイヤーの描画中にエラーが発生しました：{e}")


    # ラインレイヤー
    def line_plot(self, lines_df):
        """各統計量の線の描画"""
        try:
            # 各線を描画
            lines = alt.Chart(lines_df).mark_rule().encode(
                x="Value:Q",
                color=alt.Color(
                    "Label:N",  # 色を直接DataFrameのColor列から取得
                    scale=alt.Scale(
                        domain=["Median", "Mean", "+1σ", "-1σ"],
                        range=["green", "red", "darkorange", "darkorange"]
                    ),
                    legend=alt.Legend(
                        title="Lines",
                        orient="top-right",
                        titleAnchor="middle",
                        padding=5,
                        offset=10,
                        fillColor="rgba(200, 200, 200, 0.05)"
                        ),  # 凡例を追加
                ),
                size=alt.Size(
                    "Width:Q",  # 線の幅もDataFrameのWidth列から取得
                    legend=None  # 幅の凡例は不要
                )
            )
            return lines
        except Exception as e:
            st.error(f"ラインレイヤーの描画中にエラーが発生しました：{e}")


    # リッジラインレイヤー
    def ridge_plot(self, df):
        """リッジラインの描画"""
        try:
            chart = alt.Chart(df).mark_area(fillOpacity=0.2, interpolate='basis').encode(
                x=alt.X('x:Q', title="Value range"),
                y=alt.Y('Density:Q', title="Density", axis=alt.Axis()),
                tooltip=["x", "Density"]
            ).properties(width=600, height=400)
            return chart
        except Exception as e:
            st.error(f"リッジラインレイヤーの描画中にエラーが発生しました：{e}")


    # 折れ線グラフ
    def linechart_plot(self, df, datetime=True, col=None, dtype="N"):
        """
        折れ線グラフの描画関数
        Args:
            df (pd.DataFrame): データフレーム
            datetime_col (bool): Trueならdatetime形式を想定
            x_col (str): x軸に使用するカラム名
            x_type (str): x軸データの型 ("N", "Q", "T")
        """
        try:
            # datetime(日付)が含まれるデータの処理
            if datetime:
                x_col = "datetime"
                dtype = "T"
                # 必須カラムの存在チェック
                required_columns = ["datetime", "Value", "Category"]
                for col in required_columns:
                    if col not in df.columns:
                        raise ValueError(f"データフレームに '{col}' カラムがありません。")

                chart_base = alt.Chart(df).mark_line().encode(
                    x=alt.X(f'{x_col}:{dtype}', title="DateTime"),
                    y=alt.Y('Value:Q', title="Value"),
                    tooltip=["Value", x_col]
                ).properties(width=800, height=400)

                # Categoryが1つの場合
                if len(df["Category"].unique()) == 1:
                    chart = chart_base.encode(
                        color=alt.value('blue')
                    )

                # Categoryが複数の場合
                else:
                    chart = chart_base.encode(
                        color=alt.Color(
                            "Category:N",
                            scale=alt.Scale(scheme="tableau10"),
                            legend=alt.Legend(
                                title="Category",
                                orient="top-right",
                                fillColor="rgba(200, 200, 200, 0.05)"
                            )
                        )
                    )

                return chart


            # colに日付データがない場合(datetime_col=False)，カラムはなんでもいい
            else:
                if col is None:
                    df = df.reset_index()
                    x_col = df.columns[0]     # 元のdfのインデックスを取得
                    dtype = dtype
                else:
                    x_col = col
                    dtype=dtype

                # 共通
                chart_base = alt.Chart(df).mark_line().encode(
                    x=alt.X(f"{col}:{dtype}", title=f"{col}"),
                    y=alt.Y('Value:Q', title="Value"),
                    tooltip=["Value", f"{x_col}", "Category"]
                ).properties(width=800, height=400)


                # Categoryが1つの場合
                if len(df["Category"].unique()) == 1:
                    chart = chart_base.encode(
                        color=alt.value('blue')
                    )

                # Categoryが複数の場合
                else:
                    chart = chart_base.encode(
                        color=alt.Color(
                            "Category:N",
                            scale=alt.Scale(scheme="tableau10"),
                            legend=alt.Legend(
                                title="Category",
                                orient="top-right",
                                fillColor="rgba(200, 200, 200, 0.05)"
                            )
                        )
                    )

                return chart

        except ValueError as ve:
            st.error(f"入力データに問題があります: {ve}")
        except Exception as e:
            st.error(f"折れ線グラフレイヤーの描画中にエラーが発生しました：{e}")




