import re
import io
import zipfile
import tempfile
import random
import time
from datetime import datetime
import const
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.stats import norm
import streamlit as st
import subprocess
import altair as alt
from data_processing.base_data_processor import AWSDataProcessor, ODTDataProcessor, AMDDataProcessor, DataVisualizer
from data_processing.calculate_data import CalculateData, ApplyCondition
from altair import limit_rows, to_values
from data_processing import utils as us
import toolz
t = lambda data: toolz.curried.pipe(data, limit_rows(max_rows=10000), to_values)
alt.data_transformers.register('custom', t)
alt.data_transformers.enable('custom')

# ブラウザタブ等の設定
st.set_page_config(
    page_title="AWSデータ抽出マン",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# ページタイトル
st.title("AWSデータ抽出マン🚀（仮）")
st.write("AWS，おんどとりのデータを整理したり計算したりするよ")

# サイドバーの設定
st.sidebar.header("AWSデータ抽出マン🚀")
page = st.sidebar.radio("Page", [
    "データ整理",
    "おんどとり",
    "PygWalker"
])


# ダウンロードcsvをdfに変換する関数
@st.cache_data
def convert_df(file):
    return pd.read_csv(file, engine="python", encoding="shift-jis", index_col=0)

# ダウンロード用にdfをcsvに変換する関数
@st.cache_data
def convert_csv(df):
    return df.to_csv(index=True).encode('shift-jis')

# zipファイルを作成する関数
@st.cache_data
def create_zip(df_dict):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for name, df in df_dict.items():
            csv_data = convert_csv(df)
            zip_file.writestr(f"{name}.csv", csv_data)
    buffer.seek(0)
    return buffer

# データ処理関数を定義
@st.cache_data
def process_data(file):
    file_name = file.name
    processors = {
        "aws": AWSDataProcessor,
        "odt": ODTDataProcessor,
        "amd": AMDDataProcessor
    }
    # ファイル名の識別と処理
    for key, ProcessorClass in processors.items():
        if key in file_name.lower():
            processor = ProcessorClass(file)
            processor.load_data()
            return processor

    raise ValueError("対応するデータ形式が見つかりません")

# datatime.indexのdfをlong_dfに変換する関数
def convert_long(df):
    tmp = df.reset_index()
    long_df = tmp.melt(
                id_vars=["datetime"],
                value_vars=df.columns,
                var_name="Category",
                value_name="Value"
                )
    return long_df


# データ整理ページ
def data_page():
    # ファイルの選択
    file = st.file_uploader(
        ":material/Search: csvファイルをアップロード (ファイル名はread_me読んでね)",
        type=["csv"]
        )

    #  ページにタブを追加
    tab_data, tab_calc, tab_graphic = st.tabs(["Data", "Calculate", "Graphic"])

    # Dataタブ
    with tab_data:
        if file:
            try:
                # データ処理
                file_name = file.name
                processor = process_data(file)
                df = processor.df
                # 初期処理データの表示
                if st.checkbox("初期処理されたデータを表示"):
                    st.write(df)

                # セレクトボックスでカラムを選択
                available_columns = processor.df.columns.tolist()
                selected_columns = st.pills(
                    ":material/coffee: リサンプリングするカラムを選択",
                    available_columns,
                    selection_mode="multi")


                # リサンプリング頻度の処理
                freq_options_display = {
                    "30min": "30min",
                    "D": "daily",
                    "W": "weekly",
                    "M": "monthly",
                    "Q": "quarterly",
                    "Y": "yearly"
                }
                # リサンプリング頻度の選択
                selected_freq_display = st.pills(":material/book: リサンプリング頻度を選択", freq_options_display.values())
                # 選択された頻度をリストに変換
                selected_freq_list = next((key for key, value in freq_options_display.items() if value == selected_freq_display), None)
                # st.write(selected_freq_list)

                # 集計方法
                agg_method = {
                    "mean": "平均値",
                    "median": "中央値",
                    "std": "標準偏差",
                    "min": "最小値",
                    "max": "最大値",
                    "sum": "合計値"
                }
                # 集計方法の選択
                selected_method = st.pills(":material/key: 集計方法を選択", agg_method.values())
                # 選択された頻度をリストに変換
                selected_method_list = next((key for key, value in agg_method.items() if value == selected_method), None)
                # st.write(selected_method_list)


                # リサンプリングの設定を作成
                column_settings = {}
                for column in selected_columns:
                    column_settings[column] = {
                        "freq": selected_freq_list,
                        "method": selected_method_list
                    }
                # st.write("現在の設定")
                # st.write(f"{column_settings}")

                # リサンプリングの実行ボタンとリセットボタンを並べる
                col1, col2 = st.columns(2)
                # リサンプリングの実行と実行結果の保存
                with col1:
                    if st.button("リサンプリングの実行と保存"):
                        try:
                            if selected_columns:
                            # リサンプリングの実行
                                resampled_dfs = processor.resample_data(column_settings)    # resampled_dfs = {key:df}
                                # st.write(resampled_dfs)     # 結果の確認
                                # 結果をセッション状態に保存
                                if "resampled_dfs" not in st.session_state:
                                    st.session_state["resampled_dfs"] = {}
                                # 結果をsession_stateに追記または上書き
                                st.session_state["resampled_dfs"].update(resampled_dfs)
                                st.success("処理結果を保存しました！")
                        except ValueError as e:
                            st.error(f"一度リセットしてください：{e}")
                        except Exception as e:
                                st.error(f"{column}のリサンプリング処理中にエラーが発生しました： {e}")

                # リセットボタン
                with col2:
                    if st.button("処理結果をリセット"):
                        if "resampled_dfs" in st.session_state:
                            del st.session_state["resampled_dfs"]
                            st.success("処理結果をリセットしました")

                # 処理結果の表示
                if st.button("処理結果を表示"):
                    if "resampled_dfs" in st.session_state and st.session_state["resampled_dfs"]:
                        try:
                            # リサンプルされたDataFrameをconcat結合する
                            concat_df = pd.concat(st.session_state["resampled_dfs"].values(), axis=1)
                            # Altair用にconcat_dfを変形
                            long_df = convert_long(concat_df)
                            # session_stateにconcat_dfとlong_dfを保存
                            st.session_state["concat_df"] = concat_df
                            st.session_state["long_df"] = long_df
                        except Exception as e:
                            st.error(f"concat中にエラーが発生しました： {e}")
                    else:
                        st.error("先にリサンプリングを実行してください")

                # concat_dfとlong_dfがsession_stateにあれば，読み込んでtabを生成
                if "concat_df" in st.session_state and "long_df" in st.session_state:
                    concat_df = st.session_state["concat_df"]
                    long_df = st.session_state["long_df"]
                    # data, describeタブを生成
                    tab1, tab2 = st.tabs(["Data", "Describe"])
                    # DataVisualizerのインスタンスを生成
                    visualizer = DataVisualizer()

                    # Dataタブ
                    with tab1:
                        col1, col2 = st.columns([45, 55])
                        with col1:
                            # concat_fを表示
                            st.write(concat_df)
                            csv1 = convert_csv(concat_df)   # df更新後をcsvに変換
                            # csvのダウンロードボタン
                            st.download_button(
                                label="Download as csv",
                                data=csv1,
                                file_name=f"{file_name}_concat.csv",
                                mime="text/csv"
                                )
                            # long_dfを表示
                            st.write(long_df)
                            csv2 = convert_csv(long_df)     # df更新後をcsvに変換
                            st.download_button(
                                label="Download as csv",
                                data=csv2,
                                file_name=f"{file_name}_long.csv",
                                mime="text/csv"
                            )
                        with col2:
                            # 回帰分析の結果を計算してキャッシュに保存
                            @st.cache_data
                            def regression_stats(df):
                                res = linregress(df[x_category], df[y_category])
                                res_df = pd.DataFrame({
                                    "Metric": ["r-value", "R-squared", "slope", "intercept", "p-value", "std-error", "intercept-stderror"],
                                    "Value": [
                                        res.rvalue,
                                        res.rvalue**2,
                                        res.slope,
                                        res.intercept,
                                        res.pvalue,
                                        res.stderr,
                                        res.intercept_stderr
                                    ]
                                })
                                return res_df

                            # ユニークなカテゴリを取得
                            categories = concat_df.columns.unique()
                            # プロットするカテゴリを選択
                            if len(categories) == 1:
                                linechart = visualizer.linechart_plot(long_df)
                                st.altair_chart(linechart.interactive())
                            else:
                                # x軸のカテゴリを選択
                                x_category = st.selectbox("X-Axis", categories, key="x_category")
                                # x軸で選択されたカラムをy軸から除外
                                y_categories = [cat for cat in categories if cat != x_category]
                                y_category = st.selectbox("Y-Axis", y_categories, key="y_category")
                                # 選択された軸の軸幅を計算して，axis_rangeに保存
                                axis_range = {
                                    "x_min": float(concat_df[x_category].min())-abs(float(concat_df[x_category].min())*0.25),
                                    "x_max": float(concat_df[x_category].max())+abs(float(concat_df[x_category].min())*0.25),
                                    "y_min": float(concat_df[y_category].min())-abs(float(concat_df[y_category].min())*0.25),
                                    "y_max": float(concat_df[y_category].max())+abs(float(concat_df[y_category].min())*0.25)
                                }
                                # x軸の範囲を選択するスライダー
                                x_range = st.slider(
                                    "X-range",
                                    min_value=axis_range["x_min"],
                                    max_value=axis_range["x_max"],
                                    value=(axis_range["x_min"], axis_range["x_max"])
                                )
                                # y軸の範囲を選択するスライダー
                                y_range = st.slider(
                                    "Y-range",
                                    min_value=axis_range["y_min"],
                                    max_value=axis_range["y_max"],
                                    value=(axis_range["y_min"], axis_range["y_max"])
                                )

                                # 選択されたカテゴリをdfとして返す
                                tmp = concat_df[[f"{x_category}", f"{y_category}"]].reset_index()
                                col1, col2 = st.columns([72, 28])
                                # 散布図の描画
                                with col1:
                                    scatter_layer = visualizer.scatter_plot(tmp, x_category, y_category, x_range, y_range)
                                    st.altair_chart(scatter_layer)

                                with col2:
                                    # 回帰分析の結果を表示
                                    reg_df = regression_stats(concat_df)
                                    st.table(reg_df)

                    # Describeタブ
                    with tab2:
                        col1, col2 = st.columns([40, 60])
                        with col1:
                            st.write(concat_df.describe())  # dfの統計情報を表示
                        with col2:
                            # ヒストグラムを計算する関数
                            @st.cache_data
                            def calc_hist(series, bins):
                                """ヒストグラムのデータを計算してキャッシュ"""
                                hist, edges = np.histogram(series, bins=bins)
                                return pd.DataFrame({
                                    'count': hist,
                                    'bin_start': edges[:-1],
                                    'bin_end': edges[1:]
                                })
                            # 正規分布を計算する関数
                            @st.cache_data
                            def calc_norm(series, category):
                                """正規分布のデータを計算してキャッシュ"""
                                # Seriesから平均μ，標準偏差σを取得
                                mu, sigma = series.mean(), series.std()
                                # 確率密度関数
                                x = np.linspace(series.min(), series.max(), 100)
                                pdf = norm.pdf(x, mu, sigma)
                                # カテゴリを追加
                                df = pd.DataFrame({
                                    "category": category,
                                    "x": x,
                                    "Density":pdf
                                })
                                return df

                            # ユニークなカテゴリ名を取得してタブを生成
                            categories = list(long_df["Category"].unique()) + ["box_plot"]
                            tabs = st.tabs(categories)

                            # カテゴリごとにヒストグラムを描画
                            for i, category in enumerate(categories):
                                with tabs[i]:
                                    if category != "box_plot":
                                        # 該当カテゴリのデータをフィルタリング
                                        category_df = long_df[long_df["Category"] == category]
                                        # ヒストグラムのビンの数を選択
                                        default_bins = 30
                                        select_bins = st.slider(
                                            "ビン数を選択してください",
                                            min_value=10,
                                            max_value=100,
                                            step=5,
                                            value=default_bins,
                                            key=f"slider_{category}"
                                            )
                                        # ヒストグラムのデータを計算
                                        hist_df = calc_hist(category_df["Value"], select_bins)
                                        # 正規分布のデータを計算
                                        norm_df = calc_norm(category_df["Value"], category)
                                        # 統計量を取得
                                        mu = category_df["Value"].mean()
                                        med = category_df["Value"].median()
                                        sigma = category_df["Value"].std()

                                        # ラインレイヤーの情報をまとめる
                                        lines_df = pd.DataFrame({
                                            "Label": ["Median", "Mean", "+1σ", "-1σ"],
                                            "Value": [med, mu, mu+sigma, mu-sigma],
                                            "Color": ["red", "green", "yellow", "yellow"],
                                            "Width": [0.5, 0.5, 0.3, 0.3]
                                        })

                                        # ラインレイヤーを生成
                                        lines = visualizer.line_plot(lines_df)

                                        # グラフレイヤーを生成
                                        norm_layer = visualizer.ridge_plot(norm_df)  # リッジラインレイヤー
                                        hist_layer = visualizer.hist_plot(hist_df, select_bins)    # ヒストグラムレイヤー
                                        # ヒストグラムとリッジラインレイヤーを圧縮
                                        chart_plots = alt.layer(hist_layer, norm_layer).resolve_scale(y='independent').properties(
                                            width=600,
                                            height=400
                                        )
                                        # レイヤーを圧縮
                                        chart = chart_plots + lines
                                        # 箱ひげ図のデータ作成
                                        box_layer = visualizer.box_plot(category_df)
                                        chart = alt.hconcat(
                                            chart,
                                            box_layer
                                        ).resolve_scale(color="independent")
                                        st.altair_chart(chart, use_container_width=True)

                                    else:
                                        # box_plotタブにカテゴリごとに箱ひげ図を描画
                                        box_plot = visualizer.box_plot(long_df)
                                        st.altair_chart(box_plot, use_container_width=True)
            except Exception as e:
                    print(f"エラー：{e}")

        else:
            st.warning("解析するファイルを選択してください")


    # calculateタブ
    with tab_calc:
        # データの処理
        if file:
            # session_stateにあるdataframeを読み込み
            if "concat_df" in st.session_state and "long_df" in st.session_state:
                concat_df = st.session_state["concat_df"]
                long_df = st.session_state["long_df"]
                # 計算するカテゴリを選択
                calc_dict = {
                    "移動平均": "MA",
                    "条件判定": "JUDGE",
                    
                }
                st.selectbox(
                    "計算するカテゴリを選択してください",[
                        "移動平均",
                        "条件判定"
                        ]
                    )
            else:
                pass

        else:
            st.warning("解析するファイルを選択してください")


    # Graphicタブ
    with tab_graphic:
        if file:
            # session_stateにあるdataframeを読み込み
            if "concat_df" in st.session_state:
                concat_df = st.session_state["concat_df"]
                # Pygwalkerの起動
                st.write("💹PygWalkerでグラフ作成")
                if st.button("PygWalkerを開く"):
                    # 現在時刻と乱数でtmpファイル名を作成
                    suffix = f"{int(time.time())}_{random.randint(0, 9999)}"
                    tmp_file_name = f"pyg_config_{suffix}.pyg"
                    # データを一時ファイルに保存
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                        tmp_file.write(convert_csv(concat_df))
                        tmp_fpath = tmp_file.name
                    # Pygの起動
                    subprocess.Popen(["streamlit", "run", "pygwalker_app.py", "--", tmp_fpath])
            else:
                st.warning("セッションデータが存在しません。データをロードしてください。")
        else:
            st.warning("解析するファイルを選択してください。")


# おんどとりページの処理
def ondotori_page():
    # DataVisualizerクラスのインスタンスを生成
    visualizer = DataVisualizer()
    # ファイルの選択（複数可）
    files = st.file_uploader(
        ":material/Search: csvファイルをアップロード（複数可）",
        type=["csv"],
        accept_multiple_files=True
    )
    # 各ファイルの処理
    # おんどとりデータの初期処理を実行する関数
    @st.cache_data
    def odt_process(file):
        processor = ODTDataProcessor(file)
        processor.load_data()
        return processor

    # リサンプルする関数
    @st.cache_data
    def resample_df(df, freq_key, method_key):
        df = df.resample(freq_key).agg(method_key)
        df = df.apply(pd.to_numeric, errors='coerce')     # 数値に変換
        df = df.dropna(how='any')  # 欠損値を含む行を削除
        return df

    # リサンプルしてmax, minを取得する関数
    @st.cache_data
    def get_maxmin_df(df, freq_key):
        columns_name = df.columns
        new_df = pd.DataFrame(index=df.resample(freq_key).mean().index)
        for col in columns_name:
            new_df[f"{col}_max"] = df[col].resample(freq_key).agg(max)
            new_df[f"{col}_min"] = df[col].resample(freq_key).agg(min)
        new_df = new_df.apply(pd.to_numeric, errors='coerce')
        new_df = new_df.dropna(how='any')
        return new_df

    # タブの生成
    tab_data, tab_tempcurve, tab_fi, tab_ftjudge = st.tabs(["Data", "Temp curve", "Freezing Index", "FT judge"])

    # Dataタブ
    with tab_data:
        if files:
            df = None   # dfを初期化
            # session_stateの初期化
            if "odt_processed_df" in st.session_state and "odt_processed_df_resample" in st.session_state:
                del st.session_state["odt_processed_df"]
                del st.session_state["odt_processed_df_resample"]

            # 初期処理が必要な場合
            st.write("初期処理🐸")
            if st.checkbox("いる"):
                processed_dfs = []

                for idx, file in enumerate(files):
                    # ODTDataProcessorで初期処理
                    processor = odt_process(file)
                    df = processor.df

                    try:
                        # ファイル名に "地点_深度" が含まれる場合、その情報をカラム名に反映
                        match = re.search(r".*_([^_]*)_([^_]*)\.csv$", file.name)
                        if match:
                            prefix = match.group(1)  # 地点を抽出（例：hkb, kyrgys）
                            depth = match.group(2)  # 深度を取得（例：~cm）
                            df.columns = [f"{prefix}_{depth}" for col in df.columns]
                        else:
                            # カラム名にとインデックスを使ってユニークなカラム名を生成
                            df.columns = [f"{col}_{idx+1}" for col in df.columns]

                        processed_dfs.append(df)

                    except Exception as e:
                        st.error(f"カラム名の取得に失敗しました: {e}")

                # 横方向に結合
                if processed_dfs:
                    try:
                        df = pd.concat(processed_dfs, axis=1)
                        st.session_state["odt_processed_df"] = df
                        long_df = convert_long(df)  # long_dfに変換
                        #st.write(df)
                    except ValueError as e:
                        st.error(f"DataFrameの結合中にエラーが発生しました：{e}")
                        e = str(e)
                        if "Duplicate column names" in e:
                            st.error(
                "同じ地点や深度のデータが含まれている可能性があります．"
                "該当ファイルを確認し，削除するかファイル名を変更してください．"
            )


            # 初期処理がいらない場合
            elif st.checkbox("いらない"):
                for idx, file in enumerate(files):
                    # csvファイルの読み込み
                    df = convert_df(file)
                    # 日付インデックスに指定
                    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
                    df.dropna(subset=["datetime"], inplace=True)
                    df.set_index("datetime", inplace=True)
                    df = df.apply(pd.to_numeric, errors='coerce').dropna()

                    try:
                        # ファイル名に "地点_深度" が含まれる場合、その情報をカラム名に反映
                        match = re.search(r".*_([^_]*)_([^_]*)\.csv$", file.name)
                        if match:
                            prefix = match.group(1)  # 地点を抽出（例：hkb, kyrgys）
                            depth = match.group(2)  # 深度を取得（例：~cm）
                            df.columns = [f"{prefix}_{depth}" for col in df.columns]
                        else:
                            # カラム名にとインデックスを使ってユニークなカラム名を生成
                            df.columns = [f"{col}_{idx+1}" for col in df.columns]

                        processed_dfs.append(df)

                    except Exception as e:
                        st.error(f"カラム名の取得に失敗しました: {e}")

                # 横方向に結合
                if processed_dfs:
                    try:
                        df = pd.concat(processed_dfs, axis=1)
                        st.session_state["odt_processed_df"] = df
                        long_df = convert_long(df)  # long_dfに変換
                        st.write(df)
                    except ValueError as e:
                        st.error(f"DataFrameの結合中にエラーが発生しました：{e}")
                        e = str(e)
                        if "Duplicate column names" in e:
                            st.error(
                "同じ地点や深度のデータが含まれている可能性があります．"
                "該当ファイルを確認し，削除するかファイル名を変更してください．"
            )

            # dfが定義された場合，リサンプル処理の有無を選択
            if df is not None:
                ans = st.radio("リサンプリング", ["する", "しない"])
                if ans == "する":
                    # リサンプリング頻度の処理
                    freq_options_display = {
                        "30min": "30min",
                        "H": "hourly",
                        "D": "daily",
                        "W": "weekly",
                        "M": "monthly",
                        "Q": "quarterly",
                        "Y": "yearly"
                    }
                    # リサンプリング頻度の選択
                    selected_freq_display = st.pills(":material/book: リサンプリング頻度を選択", freq_options_display.values())
                    selected_freq_key = next((key for key, value in freq_options_display.items() if value == selected_freq_display), None)

                    # 集計方法
                    agg_method = {
                        "mean": "平均値",
                        "median": "中央値",
                        "std": "標準偏差",
                        "min": "最小値",
                        "max": "最大値",
                        "sum": "合計値"
                    }
                    # 集計方法の選択
                    selected_method = st.pills(":material/key: 集計方法を選択", agg_method.values())
                    selected_method_key = next((key for key, value in agg_method.items() if value == selected_method), None)

                    # リサンプル処理
                    if selected_freq_key is not None and selected_method_key is not None:
                        resampled_df = resample_df(df, selected_freq_key, selected_method_key)  # リサンプルの実行
                        st.session_state["odt_processed_df_resample"] = resampled_df
                        resampled_long_df = convert_long(resampled_df)
                        # col_data, col_graphに分けて実行結果を表示
                        col_data, col_graph = st.columns([45, 55])

                        # dfを表示するカラム
                        with col_data:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(resampled_df.head(100))
                                # resampled_dfをcsvに変換してDL
                                csv1 = convert_csv(resampled_df)
                                st.download_button(
                                    label="Download as csv",
                                    data=csv1,
                                    file_name=f"odt_{selected_method_key}_{selected_freq_key}.csv",
                                    mime="text/csv"
                                )
                                # long_dfに変換
                                csv2 = convert_csv(resampled_long_df)
                                st.write(resampled_long_df.head(100))
                                st.download_button(
                                    label="Download as csv",
                                    data=csv2,
                                    file_name=f"odt_{selected_method_key}_{selected_freq_key}_long.csv"
                                )
                            with col2:
                                st.write(resampled_df.describe())

                        # dfの情報をグラフ化するカラム
                        with col_graph:
                            tab1, tab2 = st.tabs(["linechart_plot", "box_plot"])
                            # 折れ線グラフの表示
                            with tab1:
                                linechart = visualizer.linechart_plot(resampled_long_df).properties(width=650, height=401)
                                st.altair_chart(linechart.interactive())
                            # 箱ひげ図の表示
                            with tab2:
                                boxplot = visualizer.box_plot(resampled_long_df)
                                st.altair_chart(boxplot)
                    else:
                        pass

                # リサンプル処理しない場合
                elif ans == "しない":
                    col1, col2 = st.columns(2)

                    with col1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(df)
                            csv2 = convert_csv(df)
                            st.download_button(
                                label="Download as csv",
                                data=csv2,
                                file_name="odt_procesed.csv",
                                mime="text/csv"
                            )
                        with col2:
                            st.write(df.describe())

                    with col2:
                        pass

        else:
            st.warning("ファイルをアップロードしてください")


    # clacタブ
    with tab_tempcurve:
        if files:
            # session_stateから処理済みのdfを呼び出す
            if "odt_processed_df" in st.session_state:
                df = st.session_state["odt_processed_df"]
                df = resample_df(df, "D", "mean")
                col1, col2 = st.columns([35, 65])
                with col1:
                    # 移動平均線の計算
                    st.write("©️移動平均線の計算")
                    # 日付範囲を取得
                    start_date = df.index.min().to_pydatetime()
                    end_date = df.index.max().to_pydatetime()

                    # 初期値を1/3の期間にする処理（グラフ描画が重いから）
                    d_length = end_date - start_date    # timedelta
                    d_lenght = d_length / 3
                    d_value = start_date + d_lenght     # 位置を計算

                    # 計算の開始日を指定
                    start = st.date_input(
                        "開始日",
                        value=start_date,
                        min_value=start_date,
                        max_value=end_date
                    )
                    # 計算の終了日を指定
                    end = st.date_input(
                        "終了日",
                        value=d_value,
                        min_value=start_date,
                        max_value=end_date
                    )

                    # dfを計算期間でフィルタリング
                    df = df[start : end]
                    long_df = convert_long(df)
                    # 移動平均線の計算条件の選択
                    col3, col4, col5 = st.columns([1, 1, 1])
                    with col3:
                        num1 = st.number_input("MA-1", step=1, min_value=1, max_value=366, value=5)
                    with col4:
                        num2 = st.number_input("MA-2", step=1, min_value=1, max_value=366, value=30)
                    with col5:
                        num3 = st.number_input("MA-3", step=1, min_value=1, max_value=366, value=100)
                    # 選択された条件をlistに格納
                    MA_list = [num1, num2, num3]
                    # 移動平均を計算して新しいdfに保存
                    @st.cache_data
                    def calc_MAs(df, ma_list):
                        """MAとstep_averageを計算してdictで返す"""
                        ma_dfs = {}    # ma_dfを格納するdict
                        step_averages = {}  # step_averageを格納するdict
                        original_columns = df.columns   # 元のカラム名を保存
                        for ma in ma_list:
                            calcdata = CalculateData(df)    # CalculateDataのインスタンスを生成
                            # 新しく移動平均を計算したdfを作成
                            ma_df = calcdata.calc_MA(ma)
                            # step間隔の平均値を計算して新しいdfに保存
                            step_average = calcdata.step_average(ma)

                            # カラム名にMA情報を追加
                            columns_name_ma = {col: f"{col}_MA{ma}" for col in original_columns}
                            columns_name_step = {col: f"{col}_step{ma}" for col in original_columns}
                            ma_df = ma_df.rename(columns=columns_name_ma)
                            step_average = step_average.rename(columns=columns_name_step)
                            # dictに保存
                            ma_dfs[f"MA{ma}"] = ma_df
                            step_averages[f"step{ma}"] = step_average
                        return ma_dfs, step_averages


                    # 移動平均線の計算
                    MAs, step_averages = calc_MAs(df, MA_list)
                    # long_dfに変換
                    MAs_long = {}   # 移動平均線
                    for key, df in MAs.items():
                        MA_long_df = convert_long(df)
                        MAs_long[key] = MA_long_df
                    step_averages_long = {}     # step間隔平均
                    for key, df in step_averages.items():
                        step_averages_long_df = convert_long(df)
                        step_averages_long[key] = step_averages_long_df


                    # 温度変化曲線の計算
                    temp_curves = {}
                    for key, ma_df in MAs.items():
                        calcdata = CalculateData(ma_df)
                        temp_curve = calcdata.temp_curve()
                        temp_curves[key] = temp_curve

                    # 年数ごとにtemp_curveを結合する     {MA5: {2021: temp_curve_df(datetime, hkb_5cm_MA5)}}←データ構造
                    temp_curve_dfs = {}    # temp_curve(df)を年数別で格納するdict
                    for key, year_data in temp_curves.items():
                        #st.write(key)
                        #st.write(year_data)
                        for year, df in year_data.items():
                            #st.write(year)
                            #st.write(df)
                            if year not in temp_curve_dfs:
                                # 年ごとのdfを作成（ここに結合していく）
                                temp_curve_dfs[year] = pd.DataFrame()
                            # yaerのカラムを作成
                            temp_curve_dfs[year]["year"] = year
                            # 1~365まで識別する
                            temp_curve_dfs[year]["Date"] = range(1, len(df) + 1)
                            # MAごとにデータを結合(dictで年別に整理してたdfを年ごとのdfとして分けて管理，つまりdictから分離)
                            temp_curve_dfs[year] = pd.concat([temp_curve_dfs[year], df], axis=1)

                    # 最後に全部縦にくっつける
                    temp_curve_result = pd.DataFrame()
                    for key, df in temp_curve_dfs.items():
                        temp_curve_result = pd.concat([temp_curve_result, df], axis=0)

                    #st.write(temp_curve_result)
                    key_year = list(temp_curve_dfs.keys())
                    #st.write(temp_curve_dfs[key_year[0]])

                    # long_dfに変換
                    temp_curve_result_long = temp_curve_result.melt(
                        id_vars=["year", "Date"],
                        value_vars=temp_curve_result.columns,
                        var_name="Category",
                        value_name="Value"
                    )
                    #st.write(temp_curve_result_long)
                    #st.write(temp_curve_result_long.columns)
                    temp_curve_long_dfs = {}
                    for year, df in temp_curve_dfs.items():
                        temp_curve_long_dfs[year] = df.melt(
                            id_vars=["year", "Date"],
                            value_vars=df.columns,
                            var_name="Category",
                            value_name="Value"
                        )

                    # MAsとtempcurve_dfsを結合（dict）
                    result_to_zip = dict(**MAs, **step_averages, **temp_curve_dfs)
                    result_to_zip_long = dict(**MAs_long, **step_averages_long, **temp_curve_long_dfs)
                    # zipファイルの作成
                    zip_buffer1 = create_zip(result_to_zip)
                    zip_buffer2 = create_zip(result_to_zip_long)
                    # ダウンロード
                    if zip_buffer1:
                        st.write("計算結果をダウンロード")
                        st.download_button(
                            "Download as zip",
                            data=zip_buffer1,
                            file_name="temp_curves.zip",
                            mime="application/zip"
                            )
                    else:
                        st.warning("計算に失敗しました")


                # グラフの描画
                with col2:
                    # 温度変化曲線の描画
                    # 描画するカテゴリーを選択
                    tc_categories = sorted(temp_curve_result_long["Category"].unique(), reverse=True)
                    tc_selected_category = st.multiselect(
                        "描画するカテゴリを選択",
                        options=tc_categories,
                        default=tc_categories[:3],
                        max_selections=5,
                    )
                    # 描画するカテゴリが選択されたら描画
                    if tc_selected_category is not None:
                        selected_df = temp_curve_result_long[temp_curve_result_long["Category"].isin(tc_selected_category)]
                        #st.write(selected_df)
                        # セレクターを作成
                        year_selector = alt.selection_multi(
                            fields=["year"],
                            bind=alt.binding_select(options=selected_df["year"].unique()),
                            name="year"
                            #init={"year": sorted(selected_df["year"].unique())[0]}
                        )
                        chart = visualizer.linechart_plot(selected_df, datetime=False, col="Date", dtype="Q")
                        chart = chart.encode(
                            strokeDash="year:N",
                            tooltip=["year"]
                        ).add_selection(year_selector).transform_filter(year_selector)
                        st.altair_chart(chart)

                    # 移動平均線の描画
                    if st.checkbox("移動平均線の表示"):
                        # 各dfを縦に結合する
                        df_list = [long_df] + list(MAs_long.values())
                        combined_df = pd.concat(df_list, ignore_index=True)     # 縦方向に結合
                        combined_df = combined_df.sort_values("Category", ascending=False)
                        combined_df["datetime"] = pd.to_datetime(combined_df["datetime"])
                        # 描画するCategoryを選択
                        MAs_categories = sorted(combined_df["Category"].unique(), reverse=True)
                        selected_category = st.multiselect(
                            "描画するカテゴリを選択",
                            options=MAs_categories,
                            default=MAs_categories[:3],
                            max_selections=5,
                            key="select1"
                        )
                        # カテゴリが選択されたら描画
                        if selected_category is not None:
                            # 選択されたカテゴリをaltair用にフィルタリング
                            selected_df = combined_df[combined_df["Category"].isin(selected_category)]
                            chart = visualizer.linechart_plot(selected_df)
                            st.altair_chart(chart)
                        else:
                            pass


                        # step間隔平均線の描画
                        df_list2 = [long_df] + list(step_averages_long.values())
                        combined_df2 = pd.concat(df_list2, ignore_index=True)
                        combined_df2 = combined_df2.sort_values("Category", ascending=False)
                        combined_df2["datetime"] = pd.to_datetime(combined_df2["datetime"])
                        # 描画するCategoryを選択
                        step_averages_categories = sorted(combined_df2["Category"].unique(), reverse=True)
                        selected_category2 = st.multiselect(
                            "描画するカテゴリを選択",
                            options=step_averages_categories,
                            default=step_averages_categories[:3],
                            max_selections=5,
                            key="select2"
                        )
                        # カテゴリが選択されたら描画
                        if selected_category2 is not None:
                            # 選択されたカテゴリをaltair用にフィルタリング
                            selected_df2 = combined_df2[combined_df2["Category"].isin(selected_category2)]
                            chart = visualizer.linechart_plot(selected_df2)
                            st.altair_chart(chart)
                        else:
                            pass
        else:
            st.warning("ファイルをアップロードしてください")


    with tab_fi:
        if files:
            # session_stateから処理済みのdfを呼び出す
            if "odt_processed_df" in st.session_state:
                df = st.session_state["odt_processed_df"]
                df = resample_df(df, "D", "mean")
                col1, col2 = st.columns([35, 65])
                with col1:
                    # 積算寒度の計算
                    st.write("©️FI・TDDの計算")
                    # 日付範囲を取得
                    start_date = df.index.min().to_pydatetime()
                    end_date = df.index.max().to_pydatetime()

                    # 初期値を1/3の期間にする処理（グラフ描画が重いから）
                    d_length = end_date - start_date    # timedelta
                    d_lenght = d_length / 3
                    d_value = start_date + d_lenght     # 位置を計算

                    # 計算の開始日を指定
                    start_FI = st.date_input(
                        "開始日",
                        value=start_date,
                        min_value=start_date,
                        max_value=end_date,
                        key="start2"
                    )
                    # 計算の終了日を指定
                    end_FI = st.date_input(
                        "終了日",
                        value=d_value,
                        min_value=start_date,
                        max_value=end_date,
                        key="end2"
                    )
                    # dfを計算期間でフィルタリング
                    df = df[start_FI : end_FI]
                    calcdata = CalculateData(df)

                    # FIの計算
                    num = st.number_input(
                        "閾値",
                        min_value=-100.0,
                        max_value=100.0,
                        value=0.0,
                        step=0.1,
                        format="%0.1f",
                        help="m℃以下を絶対値で加算．例）m=0のとき，[-1, 0, 1, -2] → A. 3℃"
                    )
                    df_FI = calcdata.calc_FI(num)
                    #st.write(df_FI)

                    # TDDの計算
                    df_TDD = calcdata.calc_TDD()
                    #st.write(df_TDD)

                    # FIとTDDを結合して表示，その後describe
                    df_FI_TDD = pd.concat([df_FI, df_TDD], axis=1).drop_duplicates()
                    col3, col4 = st.columns([55, 45])
                    with col3:
                        st.write(df_FI_TDD)
                    with col4:
                        st.write(df_FI_TDD.describe())

                    # df, FIとTDDの結果を結合
                    df = pd.concat([df, df_FI, df_TDD], axis=1).drop_duplicates()

                    # 月ごとに集計したデータの作成
                    df_months = us.df_groups(df, freq_key="M")

                    # FIとTDDのzipファイルを作成
                    result = dict(**{"FI": df_FI, "TDD": df_TDD}, **df_months)

                    # ダウンロード
                    zip_buffer_FI = create_zip(result)
                    if zip_buffer_FI:
                        st.write("計算結果をダウンロード")
                        st.download_button(
                            "Download as zip",
                            data=zip_buffer_FI,
                            file_name="FI_TDD.zip",
                            mime="application/zip"
                            )
                    else:
                        st.warning("ダウンロードに失敗しました")


                with col2:
                    # df(FI, TDD追加済)をlong_dfに変換
                    df_long = convert_long(df_FI_TDD)
                    # カテゴリの選択
                    df_categories = sorted(df_long["Category"].unique())
                    selected_category = st.multiselect(
                        "描画するカテゴリを選択",
                        options=df_categories,
                        default=df_categories[:3],
                        max_selections=5,
                    )
                    # グラフの描画
                    if selected_category is not None:
                        visualizer = DataVisualizer()
                        selected_df = df_long[df_long["Category"].isin(selected_category)]
                        chart = visualizer.linechart_plot(selected_df)
                        st.altair_chart(chart)
                        pass


        else:
            st.warning("ファイルをアップロードしてください")


    @st.cache_data
    def apply_condition(df, freq_key):
        ac = ApplyCondition(df, freq_key=freq_key)
        return ac
    # FT judgeタブ
    with tab_ftjudge:
        if files:
            # session_stateから処理済みのdfを呼び出す
            if "odt_processed_df" in st.session_state:
                df = st.session_state["odt_processed_df"]
                col1, col2 = st.columns([35, 65])

                with col1:
                    st.write("©️凍結融解判定")

                    # 日付範囲を取得
                    start_date = df.index.min().to_pydatetime()
                    end_date = df.index.max().to_pydatetime()

                    # 初期値を1/3の期間にする処理（適当）
                    d_length = end_date - start_date    # timedelta
                    d_lenght = d_length / 3
                    d_value = start_date + d_lenght     # 位置を計算

                    # 計算の開始日を指定
                    start_judge = st.date_input(
                        "開始日",
                        value=start_date,
                        min_value=start_date,
                        max_value=end_date,
                        key="start3"
                    )
                    # 計算の終了日を指定
                    end_judge = st.date_input(
                        "終了日",
                        value=d_value,
                        min_value=start_date,
                        max_value=end_date,
                        key="end3"
                    )
                    # dfを計算期間でフィルタリング
                    df = df[start_judge : end_judge]


                    # FTjudgeの実行
                    ac = apply_condition(df, freq_key="D")
                    ft_df = ac.judge_freeze()
                    ft_total = ac.get_count()

                    # 月ごとに集計
                    result = us.df_groups(ft_df, freq_key="M")

                    col3, col4 = st.columns([60, 40])
                    with col3:
                        # カウント列を加えてdfを表示
                        st.write(us.add_count_row(ft_df))
                    with col4:
                        st.write(ft_total)

                    # ダウンロード
                    result = dict(**result, **{"total": us.add_count_row(ft_df)})
                    zip_buffer_FT = create_zip(result)
                    if zip_buffer_FT:
                        st.write("計算結果をダウンロード")
                        st.download_button(
                            "Download as zip",
                            data=zip_buffer_FT,
                            file_name="FTjudge.zip",
                            mime="application/zip"
                            )
                    else:
                        st.warning("ダウンロードに失敗しました")


                # グラフの描画
                with col2:
                    pass



# PygWalkerページの処理
def Pyg_page():
    # ファイルの選択
    file = st.file_uploader(
        ":material/Search: csvファイルをアップロード",
        type=["csv"]
        )
    if file:
        df = pd.read_csv(file, engine='python', encoding="shift-jis")
        if st.button("PygWalkerを開く"):
            # 現在時刻と乱数でtmpファイル名を作成
            suffix = f"{int(time.time())}_{random.randint(0, 9999)}"
            tmp_file_name = f"pyg_config_{suffix}.pyg"
            # データを一時ファイルに保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(convert_csv(df))
                tmp_fpath = tmp_file.name
            # Pygの起動
            subprocess.Popen(["streamlit", "run", "pygwalker_app.py", "--", tmp_fpath])
    else:
        st.warning("解析するファイルを選択してください。")


# pageの選択と処理
if page == "データ整理":
    data_page()

# おんどとり
elif page == "おんどとり":
    ondotori_page()

# PygWalker
elif page == "PygWalker":
    Pyg_page()
else:
    pass