import os
import glob
import pandas as pd

# 作業ディレクトリの確認
print(f"Current directory: {os.getcwd()}")

# 検索する地域を入力
site = input("AWSの地域を入力して下さい（例: kyrgyz, hakubaなど）: ")

# CSVファイルの検索
pattern = f"*{site}*.csv"
matching_files = glob.glob(pattern)

# ファイルの検出
if matching_files:
    print(f"\n検索結果 \"{site}\":")
    for i, file in enumerate(matching_files):
        print(f"{i}: {file}")
    try:
        choice = int(input("ファイルの数字を選択してね: "))
        selected_file = matching_files[choice]
        print(f"Loading file: {selected_file}")
        df = pd.read_csv(selected_file, skiprows=[0, 1, 2, 4, 5], encoding="shift-jis", low_memory=False)
    except (IndexError, ValueError):
        print("Invalid selection. Exiting.")
        exit()
else:
    print("そんなファイルないよ.")
    exit()


# 不要列の削除
columns_drop = ["Unnamed: 1", "Unnamed: 13", "Unnamed: 14", "Unnamed: 15", 
                "Unnamed: 16", "Unnamed: 17", "Unnamed: 18", "Unnamed: 19", 
                "Unnamed: 20", "Unnamed: 21", "本体電力"]
df = df.drop(columns=columns_drop, errors='ignore')

print(df.columns)
# 列名の変更
df.rename(columns={
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


# datetime列を時系列データとして設定
df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
df = df.dropna(subset=["datetime"])
df = df.set_index("datetime")

# 数値列の変換と欠損値の削除
df = df.apply(pd.to_numeric, errors='coerce').dropna()

print(df.head(5))

# 各時間スケールでの集計
agg_funcs = {"H": "hourly", "D": "daily", "W": "weekly", "M": "monthly", "Y": "yearly"}
results = {}

for freq, label in agg_funcs.items():
    results[f"{label}_mean"] = df.resample(freq).mean()
    if "prec(mm)" in df.columns:
        results[f"{label}_sum"] = df["prec(mm)"].resample(freq).sum()

# 各スケールごとにcsv出力
for freq, label in agg_funcs.items():
    # prec以外のdf
    mean_df = results[f"{label}_mean"]
    # precのdf
    if f"{label}_sum" in results:
        sum_df = results[f"{label}_sum"]
        # precとそれ以外を結合
        combined_df = pd.concat([sum_df, mean_df], axis=1)
    else:
        combined_df = mean_df
    
    # 結果の確認
    print(f"\n--- {label} ---")
    print(combined_df.head(3))
    
    # スケールごとにCSV保存
    output_filename = f"{site}_{label}.csv"
    combined_df.to_csv(output_filename, encoding="shift-jis")
    print(f"\n--- {output_filename} を保存したよ．やったね ---")
