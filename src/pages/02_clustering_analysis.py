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

st.set_page_config(page_title="군집 분석", layout="wide")


def apply_custom_style():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 1.5rem;}
        .hero {
            border-radius: 16px; padding: 18px 20px; margin-bottom: 14px;
            background: linear-gradient(120deg, #0f766e 0%, #0891b2 55%, #2563eb 100%);
            color: #ffffff;
        }
        .hero h2 {margin: 0; font-size: 1.35rem;}
        .hero p {margin: 8px 0 0 0; opacity: 0.95;}
        .cluster-card {
            border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px 16px;
            background: #f8fafc; height: 100%;
        }
        .cluster-card h4 {margin: 0 0 8px 0;}
        .tag {
            display: inline-block; padding: 2px 9px; border-radius: 999px;
            font-size: 0.76rem; margin-right: 6px; margin-bottom: 7px;
            border: 1px solid #cbd5e1; background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(dropout_rate: float) -> str:
    if dropout_rate >= 40:
        return "이탈 고위험"
    if dropout_rate >= 25:
        return "이탈 주의"
    return "이탈 안정"


def profile_text(row: pd.Series, q_score: float, q_active: float, q_clicks: float) -> tuple[str, str]:
    engagement = "높음" if row["평균활동일수"] >= q_active and row["평균클릭수"] >= q_clicks else "보통"
    if row["평균활동일수"] < q_active * 0.7 or row["평균클릭수"] < q_clicks * 0.7:
        engagement = "낮음"

    achievement = "높음" if row["평균점수"] >= q_score else "보통"
    if row["평균점수"] < q_score * 0.75:
        achievement = "낮음"

    comment = (
        f"학습 참여도는 {engagement}, 성취도는 {achievement} 수준입니다. "
        f"이탈 위험도는 '{risk_badge(row['이탈률'])}'로 분류됩니다."
    )
    return engagement, comment


apply_custom_style()
st.markdown(
    """
    <div class="hero">
      <h2>학습자 군집 분석</h2>
      <p>군집별 학습 패턴과 이탈 위험 신호를 한눈에 확인하고, 우선 관리 대상 군집을 빠르게 찾을 수 있습니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("현재 UI는 Light 모드 기준으로 가독성과 색상 대비를 최적화했습니다.")

try:
    df_cluster = load_clusters()
except Exception as e:
    st.error(f"DB 연결 실패: {e}")
    st.stop()

if df_cluster.empty:
    st.warning("군집 데이터가 없습니다.")
    st.stop()

# 표시용 컬럼 추가
label_map = {0: "군집 A", 1: "군집 B", 2: "군집 C", 3: "군집 D", 4: "군집 E"}
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

total_students = int(summary["학생수"].sum())
dominant_cluster = summary.sort_values("학생수", ascending=False).iloc[0]["군집"]
avg_dropout_rate = float(summary["이탈률"].mean())
top_risk_cluster = summary.sort_values("이탈률", ascending=False).iloc[0]["군집"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 학생 수", f"{total_students:,}")
c2.metric("군집 수", f"{summary['군집'].nunique()}")
c3.metric("평균 이탈률", f"{avg_dropout_rate:.1f}%")
c4.metric("관리 우선 군집", top_risk_cluster)

st.dataframe(summary, use_container_width=True)

st.divider()

st.subheader("군집별 해석")
q_score = float(summary["평균점수"].median())
q_active = float(summary["평균활동일수"].median())
q_clicks = float(summary["평균클릭수"].median())
cluster_cols = st.columns(min(3, max(1, len(summary))))
for idx, (_, row) in enumerate(summary.sort_values("군집").iterrows()):
    engagement, comment = profile_text(row, q_score, q_active, q_clicks)
    with cluster_cols[idx % len(cluster_cols)]:
        st.markdown(
            f"""
            <div class="cluster-card">
              <h4>{row['군집']}</h4>
              <span class="tag">학생수 {int(row['학생수'])}명</span>
              <span class="tag">참여도 {engagement}</span>
              <span class="tag">{risk_badge(float(row['이탈률']))}</span>
              <p style="margin-top:8px; margin-bottom:0;">{comment}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("군집별 학생 비율")
    fig_pie = px.pie(
        summary,
        names="군집",
        values="학생수",
        hole=0.45,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("군집별 성취도/참여도")
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
