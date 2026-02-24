import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common_data import load_clusters

st.title("군집 분석")

try:
    df_cluster = load_clusters()
except Exception as e:
    st.error(f"DB 연결 실패: {e}")
    st.stop()

if df_cluster.empty:
    st.warning("군집 데이터가 없습니다.")
    st.stop()

# 표시용 컬럼 추가
label_map = {0: "군집 0", 1: "군집 1", 2: "군집 2", 3: "군집 3", 4: "군집 4"}
df_cluster = df_cluster.copy()
df_cluster["군집"] = df_cluster["cluster_id"].map(label_map).fillna("기타 군집")

st.subheader("군집 요약")
summary = (
    df_cluster.groupby("군집", as_index=False)
    .agg(
        학생수=("id_student", "count"),
        평균점수=("avg_score", "mean"),
        평균활동일수=("active_days", "mean"),
        평균클릭수=("total_clicks", "mean"),
        이탈률=("dropout", "mean"),
    )
)
summary["이탈률"] = (summary["이탈률"] * 100).round(1)
summary["평균점수"] = summary["평균점수"].round(1)
summary["평균활동일수"] = summary["평균활동일수"].round(1)
summary["평균클릭수"] = summary["평균클릭수"].round(1)

st.dataframe(summary, use_container_width=True)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("군집 인원 비율")
    fig_pie = px.pie(
        summary,
        names="군집",
        values="학생수",
        hole=0.45,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("군집별 평균점수/활동일수")
    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(name="평균 점수", x=summary["군집"], y=summary["평균점수"], yaxis="y1")
    )
    fig_bar.add_trace(
        go.Scatter(
            name="평균 활동일수",
            x=summary["군집"],
            y=summary["평균활동일수"],
            yaxis="y2",
            mode="lines+markers",
        )
    )
    fig_bar.update_layout(
        xaxis_title="군집",
        yaxis=dict(title="평균 점수", rangemode="tozero"),
        yaxis2=dict(title="평균 활동일수", overlaying="y", side="right", rangemode="tozero"),
        legend_title="지표",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

st.subheader("활동 대비 성적 분포")
fig_scatter = px.scatter(
    df_cluster,
    x="active_days",
    y="avg_score",
    color="군집",
    size="total_clicks",
    hover_data=["id_student", "dropout"],
    labels={
        "active_days": "활동 일수",
        "avg_score": "평균 점수",
        "total_clicks": "총 클릭 수",
        "dropout": "실제 이탈 여부",
    },
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

st.subheader("군집 프로파일 비교 (정규화)")
features = ["avg_score", "active_days", "total_clicks"]
feature_labels = ["평균 점수", "활동 일수", "총 클릭 수"]
scaler = MinMaxScaler()
cluster_means = df_cluster.groupby("군집")[features].mean()
cluster_norm = pd.DataFrame(
    scaler.fit_transform(cluster_means), columns=features, index=cluster_means.index
)

fig_radar = go.Figure()
for cluster_name in cluster_norm.index:
    r_vals = cluster_norm.loc[cluster_name].tolist()
    r_vals += [r_vals[0]]
    theta_vals = feature_labels + [feature_labels[0]]
    fig_radar.add_trace(
        go.Scatterpolar(r=r_vals, theta=theta_vals, fill="toself", name=cluster_name)
    )

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    height=450,
    legend_title="군집",
)
st.plotly_chart(fig_radar, use_container_width=True)
