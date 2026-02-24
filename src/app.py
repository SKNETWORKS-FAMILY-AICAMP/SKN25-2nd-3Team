import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="OULAD 통합 분석 대시보드", layout="wide")

pg = st.navigation(
    [
        st.Page("pages/01_dropout_prediction.py", title="이탈 예측"),
        st.Page("pages/02_clustering_analysis.py", title="군집 분석"),
    ],
    position="sidebar",
)

pg.run()
