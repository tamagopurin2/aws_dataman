# aws_dataman
<img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">

# AWSデータ抽出マン🚀（仮）

シンプル操作で **AWS（Automatic Weather Station）／おんどとり（ODT）／AMD** 系CSVを読み込み、
前処理・リサンプリング・統計要約・回帰・可視化・各種指標（FI/TDD・凍結融解判定など）を **Streamlit** 上で行うデータ整理アプリです。

> 本READMEは過去ソース（本リポジトリの `app.py` 相当）から逆引きで作成した暫定版です。実装と差分が出た場合は適宜更新してください。

---

## 目次
- [主な機能](#主な機能)
- [画面構成](#画面構成)
- [入力データ仕様](#入力データ仕様)
- [リサンプリングと集計](#リサンプリングと集計)
- [可視化（Altair）](#可視化altair)
- [計算系機能](#計算系機能)
- [ファイル出力](#ファイル出力)
- [動作要件](#動作要件)
- [セットアップ](#セットアップ)
- [起動方法](#起動方法)
- [ディレクトリ構成例](#ディレクトリ構成例)
- [よくある質問 / トラブルシュート](#よくある質問--トラブルシュート)
- [今後のTODO](#今後のtodo)
- [ライセンス](#ライセンス)

---

## 主な機能
- **CSVの読み込み & 前処理**
  - 文字コード **Shift-JIS** 前提の CSV を読み込み（`convert_df`）。
  - `data_processing` 配下の各 Processor でソース別に前処理：
    - `AWSDataProcessor` / `ODTDataProcessor` / `AMDDataProcessor`（自作モジュール）
- **リサンプリング & 集計**（列ごとに設定）
  - 周期：30min / D / W / M / Q / Y（おんどとりページでは H も）
  - 集計方法：平均 / 中央値 / 標準偏差 / 最小 / 最大 / 合計
- **統計要約 & 回帰**（データ整理ページ）
  - `scipy.stats.linregress` による単回帰（R, R², slope, intercept など）
  - `describe()` による統計量表示
- **可視化（Altair）**
  - 折れ線・散布図・ヒストグラム・リッジライン・箱ひげ図
  - 軸レンジスライダ、カテゴリ選択、年別の線種（strokeDash）などのインタラクション
- **おんどとり（ODT）支援**
  - 複数ファイルの横結合（ファイル名から地点・深度を抽出）
  - 移動平均（複数窓）・ステップ平均・温度変化曲線（年別集計）
  - **FI（Freezing Index）/ TDD（Thawing Degree Days）** 計算
  - **凍結融解（Freeze/Thaw）判定** と月次集計
- **ダウンロード**
  - 加工結果を **CSV** または **ZIP**（複数CSVまとめ）で出力

---

## 画面構成
サイドバーの `Page` 切替で遷移します。

### 1) データ整理
- **ファイルアップロード**（単一CSV）
- **前処理**：ファイル名から `aws/odt/amd` を判別し該当 Processor を適用
- **リサンプリング設定**（列×周期×集計）→ 実行・保存 → 連結表示
- **Data タブ**
  - 連結DF（`concat_df`）と Long 形式（`long_df`）の表示とCSVダウンロード
  - 散布図 + 単回帰（カテゴリをX/Yに選択、レンジスライダ、回帰統計表）
- **Describe タブ**
  - `concat_df.describe()`
  - 各カテゴリの **ヒストグラム + 正規分布（PDF） + 参照線（中央値・平均・±1σ）**
  - **カテゴリ横断の箱ひげ図**

### 2) おんどとり
- **Data タブ**
  - 複数CSVを読み込み、必要に応じて初期処理（`ODTDataProcessor`）
  - ファイル名から **地点_深度** を抽出して列名に反映
  - 任意の周期・集計でリサンプリングし、統計量・グラフ・CSV/long CSV をDL
- **Temp curve タブ**
  - 期間指定 → 複数窓の **移動平均** と **ステップ平均** を計算
  - 年別の **温度変化曲線** を生成し、カテゴリ選択・年セレクタで可視化
  - 計算結果一式を **ZIP** でDL
- **Freezing Index タブ**
  - 期間指定 → **FI** 閾値設定（例：0℃以下）と **TDD** を計算
  - 合体表・統計量表示、月次集計、ZIPダウンロード
- **FT judge タブ**
  - 期間指定 → **凍結融解判定** 実行、日次表と合計値、月次集計を表示
  - ZIPダウンロード

### 3) PygWalker
- **（将来対応予定）**

---

## 入力データ仕様
- **文字コード**：Shift-JIS（`convert_df` / `convert_csv` が SJIS を前提）
- **日時列**：`datetime`（`convert_df` 後に index に設定）
- **数値列**：`pd.to_numeric(..., errors='coerce')` 後、欠損除去
- **おんどとりの列名付与**：
  - ファイル名パターン `.*_([^_]*)_([^_]*)\.csv$` に一致すると、
    - `prefix = 地点`、`depth = 深度` を抽出し、列名を `prefix_depth` へリネーム
  - 例：`hkb_5cm.csv` → 列名 `hkb_5cm`

> 実ファイルの項目や型が異なる場合、`data_processing` 配下の Processor 実装に合わせて調整してください。

---

## リサンプリングと集計
- 周期（例）：`30min` / `H` / `D` / `W` / `M` / `Q` / `Y`
- 集計：`mean` / `median` / `std` / `min` / `max` / `sum`
- 列ごとに **頻度×集計** を指定 → 実行すると結果がセッションに蓄積
- 表示時にすべてのリサンプル結果を **列方向に結合**（`concat_df`）

---

## 可視化（Altair）
- **データ量制約の回避**：
  - `alt.data_transformers.register('custom', toolz.pipe(..., limit_rows(10000), to_values))`
  - 表示前に行数を 10k に制限して描画負荷を軽減
- **代表チャート**：
  - 折れ線（時系列 / Date 指定にも対応）
  - 散布図（X/Y 切替、レンジスライダ、ツールチップ）
  - ヒストグラム + 正規分布PDF + 参照線（Mean/Median/±1σ）
  - 箱ひげ図（カテゴリ一括 or 各カテゴリ）
  - リッジライン（分布形状の俯瞰）

---

## 計算系機能
- **単回帰**：`scipy.stats.linregress`（R, R², slope, intercept, p, stderr 等）
- **移動平均 / ステップ平均**：`CalculateData.calc_MA` / `step_average`
- **温度変化曲線**：`CalculateData.temp_curve`（年別DataFrameを生成）
- **FI / TDD**：`CalculateData.calc_FI(threshold)` / `calc_TDD()`
- **凍結融解判定**：`ApplyCondition(...).judge_freeze()` と `get_count()`
- **グルーピング支援**：`utils.df_groups(df, freq_key='M')` で月次などの集計表を作成

---

## ファイル出力
- **CSV**：各結果を **Shift-JIS** で `Download as csv`
- **ZIP**：複数結果を `create_zip({name: df, ...})` で一括DL
  - `*_long.csv`：Altair向けに縦持ちへ `melt` した形式

---

## 動作要件
- **Python**：3.10 以上推奨
- **主要ライブラリ**（抜粋）
  - `streamlit`, `pandas`, `numpy`, `scipy`, `altair`, `pygwalker`, `toolz`
  - 自作モジュール：`data_processing`（`base_data_processor.py`, `calculate_data.py`, `utils.py` など）

> 依存は `requirements.txt` を作成して固定化することを推奨します。

---

## セットアップ
```bash
# 仮想環境は任意
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate

# 依存関係のインストール（例）
pip install streamlit pandas numpy scipy altair pygwalker toolz

# ローカルモジュール（data_processing）をパス解決できるように
# プロジェクト直下をカレントディレクトリにして起動してください
```

> CSV が Shift-JIS のため、**Windows 環境での検証**も行うと安心です。

---

## 起動方法
```bash
streamlit run app.py
```
- 初回アクセス時にブラウザが開きます。サイドバーから `データ整理` / `おんどとり` / `PygWalker` を選択。

---

## ディレクトリ構成例
```
repo_root/
├─ app.py                    # 本体（Streamlit）
├─ data_processing/
│  ├─ __init__.py
│  ├─ base_data_processor.py # AWS/ODT/AMD 用 Processor と DataVisualizer
│  ├─ calculate_data.py      # FI/TDD/MA/温度変化曲線/凍結融解などの計算
│  └─ utils.py               # 集計・整形のユーティリティ
├─ requirements.txt          # （推奨）依存を固定
└─ README.md                 # 本ファイル
```

---

## よくある質問 / トラブルシュート
- **Q. CSV の読み込みで文字化けする**
  - 本アプリは **Shift-JIS** を前提にしています。UTF-8 の場合は `convert_df` / `convert_csv` のエンコーディングを調整してください。
- **Q. グラフが重い / 途中で描画が止まる**
  - 10k 行上限のトランスフォーマ（`limit_rows`）を入れていますが、さらに軽量化が必要なら**期間を短く**、または**列数を減らす**などで調整してください。
- **Q. ODT の列名が想定どおりにならない**
  - ファイル名が `..._<地点>_<深度>.csv` 形式であるか確認してください。正規表現に一致しないと連番付与になります。
- **Q. リサンプリング結果の結合でエラーが出る**
  - セッションに前回の結果が残っている可能性があります。`処理結果をリセット` ボタンでクリアしてから再実行してください。
- **Q. AMD とは？**
  - 実データ源に依存します。`AMDDataProcessor` 実装のコメントやドキュメントを確認してください（AMeDAS 等の略の可能性）。

---

## 今後のTODO
- [ ] `PygWalker` ページの実装・安定化
- [ ] 依存ライブラリのバージョン固定（`requirements.txt`）
- [ ] `data_processing` 各 Processor の入出力仕様のREADME化
- [ ] 単回帰の多変量対応 / 交互作用の簡易検定
- [ ] 大規模データのストリーミング／サンプリング最適化
- [ ] E2E テスト（`streamlit.testing`）の追加

---

## ライセンス
プロジェクトの方針に合わせて追記してください（例：MIT / Apache-2.0 など）。

