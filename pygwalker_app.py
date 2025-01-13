import streamlit as st
import pygwalker as pyg
import pandas as pd
import sys
from pygwalker.api.streamlit import StreamlitRenderer


st.set_page_config(page_title="Pygwalker", layout="wide")

def run_pygwalker(data):
    """PygWalkerの起動"""
    pyg_app = StreamlitRenderer(data)
    pyg_app.explorer()

if __name__ == "__main__":
    # コマンドライン引数からファイルパスを取得
    if len(sys.argv) > 1:
        f_path = sys.argv[1]
        try:
            # CSVファイルを読み込み
            data = pd.read_csv(f_path, encoding="shift-jis")
            run_pygwalker(data)
        except pd.errors.EmptyDataError:
            st.error("ファイルが空のようです。内容を確認してください。")
        except pd.errors.ParserError as e:
            st.error(f"ファイル形式に問題があります: {e}")
        except Exception as e:
            st.error(f"データの読み込み中に予期しないエラーが発生しました: {e}")
    else:
        st.error("データファイルのパスが正しく渡されていません")