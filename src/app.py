import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# ================================
# 기본 설정
# ================================
load_dotenv()
st.set_page_config(page_title="OULAD 통합 분석 대시보드", layout="wide")
st.title("OULAD 학습자 분석 통합 대시보드")

# ================================
# DB 연결
# ================================
@st.cache_resource
def get_engine():
    return create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

# ================================
# 데이터 로딩
# ================================
@st.cache_data(ttl=60)
def load_predictions():
    engine = get_engine()
    return pd.read_sql("""
        SELECT p.id_student, p.model_name, p.predicted, p.probability, p.run_id,
               s.gender, s.age_band, s.region, s.highest_education,
               s.imd_band, s.studied_credits, s.dropout
        FROM predictions p
        LEFT JOIN students s ON p.id_student = s.id_student
    """, engine)

@st.cache_data(ttl=60)
def load_clusters():
    engine = get_engine()
    return pd.read_sql("""
        SELECT 
            c.id_student,
            c.cluster_id,
            s.avg_score,
            s.active_days,
            s.total_clicks,
            s.dropout
        FROM clusters c
        JOIN students s ON c.id_student = s.id_student
    """, engine)

# ================================
# 메뉴 선택
# ================================
menu = st.sidebar.radio(
    "메뉴 선택",
    ["이탈 예측 분석", "군집 분석"]
)

# ================================
# 1️⃣ 이탈 예측 분석
# ================================
if menu == "이탈 예측 분석":

    try:
        df_pred = load_predictions()
    except Exception as e:
        st.error(f"DB 연결 실패: {e}")
        st.stop()

    models = df_pred["model_name"].unique().tolist()
    selected_model = st.sidebar.selectbox("모델 선택", models)

    df = df_pred[df_pred["model_name"] == selected_model].copy()

    total = len(df)
    predicted_dropout = int(df["predicted"].sum())
    actual_dropout = int(df["dropout"].sum()) if "dropout" in df.columns else None
    avg_prob = df["probability"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 학생 수", f"{total:,}")
    col2.metric("예측 이탈", f"{predicted_dropout:,}", f"{predicted_dropout/total*100:.1f}%")

    if actual_dropout is not None:
        col3.metric("실제 이탈", f"{actual_dropout:,}", f"{actual_dropout/total*100:.1f}%")

    col4.metric("평균 이탈 확률", f"{avg_prob:.3f}")

    st.divider()

    # 모델 성능 비교
    if len(models) > 1:
        st.subheader("모델별 성능 비교")

        rows = []
        for m in models:
            d = df_pred[df_pred["model_name"] == m]
            if "dropout" in d.columns and d["dropout"].notna().any():
                y_true = d["dropout"].dropna().astype(int)
                y_pred = d.loc[y_true.index, "predicted"].astype(int)
                y_prob = d.loc[y_true.index, "probability"]

                rows.append({
                    "모델": m,
                    "F1": f"{f1_score(y_true, y_pred):.4f}",
                    "AUC": f"{roc_auc_score(y_true, y_prob):.4f}",
                    "Accuracy": f"{accuracy_score(y_true, y_pred):.4f}",
                    "예측 이탈 수": int(y_pred.sum())
                })

        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("모델"), use_container_width=True)

    st.divider()

    # 확률 분포
    st.subheader("이탈 확률 분포")

    col_a, col_b = st.columns(2)

    with col_a:
        st.bar_chart(
            df["probability"].value_counts(bins=10, sort=False).sort_index()
        )

    with col_b:
        dropout_counts = df["predicted"].value_counts().rename({0: "유지", 1: "이탈"})
        st.bar_chart(dropout_counts)

    st.divider()

    # 그룹별 이탈률
    st.subheader("그룹별 이탈 예측 비율")

    group_col = st.selectbox(
        "그룹 기준",
        ["age_band", "gender", "highest_education", "imd_band", "region"]
    )

    if group_col in df.columns:
        group_df = (
            df.groupby(group_col)["predicted"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "이탈", "count": "전체"})
        )
        group_df["이탈률"] = (group_df["이탈"] / group_df["전체"] * 100).round(1)
        group_df = group_df.sort_values("이탈률", ascending=False)
        st.dataframe(group_df, use_container_width=True)

    st.divider()

    # 학생 상세 조회
    st.subheader("학생 상세 조회")

    threshold = st.slider("이탈 확률 임계값", 0.0, 1.0, 0.5, 0.05)
    high_risk = df[df["probability"] >= threshold].sort_values("probability", ascending=False)

    st.write(f"{len(high_risk):,}명 (확률 ≥ {threshold})")

    st.dataframe(
        high_risk[[
            "id_student", "probability", "predicted", "dropout",
            "gender", "age_band", "region", "studied_credits"
        ]].reset_index(drop=True),
        use_container_width=True,
        height=300
    )

# ================================
# 2️⃣ 군집 분석
# ================================
elif menu == "군집 분석":

    try:
        df_cluster = load_clusters()
    except Exception as e:
        st.error(f"DB 연결 실패: {e}")
        st.stop()

    st.subheader("군집별 행동 패턴 요약")

    summary = df_cluster.groupby('cluster_id').agg({
        'avg_score': 'mean',
        'active_days': 'mean',
        'total_clicks': 'mean'
    }).reset_index()

    cols = st.columns(len(summary))

    for i, row in summary.iterrows():
        with cols[i]:
            st.metric(
                label=f"Cluster {int(row['cluster_id'])}",
                value=f"{round(row['avg_score'],1)}점",
                delta=f"접속 {round(row['active_days'],1)}일"
            )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("군집 비율")
        st.bar_chart(df_cluster["cluster_id"].value_counts())

    with col2:
        st.subheader("활동 대비 성적 분포")
        fig = px.scatter(
            df_cluster,
            x="active_days",
            y="avg_score",
            color="cluster_id",
            size="total_clicks",
            hover_data=["id_student"]
        )
        st.plotly_chart(fig, width='stretch')

    st.divider()

    st.subheader("군집 프로파일 비교 (정규화)")

    features = ["avg_score", "active_days", "total_clicks"]
    scaler = MinMaxScaler()
    cluster_means = df_cluster.groupby("cluster_id")[features].mean()
    cluster_norm = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        columns=features,
        index=cluster_means.index
    )

    fig_radar = go.Figure()

    for cluster in cluster_norm.index:
        r_vals = cluster_norm.loc[cluster].tolist()
        r_vals += [r_vals[0]]
        theta_vals = features + [features[0]]

        fig_radar.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill='toself',
                name=f'Cluster {cluster}'
            )
        )

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400
    )

    st.plotly_chart(fig_radar, width='stretch')