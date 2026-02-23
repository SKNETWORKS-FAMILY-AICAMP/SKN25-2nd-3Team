import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="OULAD 이탈 예측 대시보드", layout="wide")

@st.cache_resource
def get_engine():
    return create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

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
def load_students():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM students", engine)

# ================================
# 헤더
# ================================
st.title("OULAD 이탈 예측 대시보드")

try:
    df_pred = load_predictions()
    df_students = load_students()
except Exception as e:
    st.error(f"DB 연결 실패: {e}")
    st.stop()

# ================================
# 사이드바 - 모델 선택
# ================================
models = df_pred["model_name"].unique().tolist()
selected_model = st.sidebar.selectbox("모델 선택", models)

df = df_pred[df_pred["model_name"] == selected_model].copy()

# ================================
# 전체 지표
# ================================
total     = len(df)
predicted_dropout = int(df["predicted"].sum())
actual_dropout    = int(df["dropout"].sum()) if "dropout" in df.columns else None
avg_prob  = df["probability"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 학생 수", f"{total:,}")
col2.metric("예측 이탈", f"{predicted_dropout:,}", f"{predicted_dropout/total*100:.1f}%")
if actual_dropout is not None:
    col3.metric("실제 이탈", f"{actual_dropout:,}", f"{actual_dropout/total*100:.1f}%")
col4.metric("평균 이탈 확률", f"{avg_prob:.3f}")

st.divider()

# ================================
# 모델별 성능 비교
# ================================
if len(models) > 1:
    st.subheader("모델별 성능 비교")

    rows = []
    for m in models:
        d = df_pred[df_pred["model_name"] == m]
        if "dropout" in d.columns and d["dropout"].notna().any():
            from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
            y_true = d["dropout"].dropna().astype(int)
            y_pred = d.loc[y_true.index, "predicted"].astype(int)
            y_prob = d.loc[y_true.index, "probability"]
            rows.append({
                "모델": m,
                "F1": f"{f1_score(y_true, y_pred):.4f}",
                "AUC": f"{roc_auc_score(y_true, y_prob):.4f}",
                "Accuracy": f"{accuracy_score(y_true, y_pred):.4f}",
                "예측 이탈 수": int(y_pred.sum()),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("모델"), use_container_width=True)

    st.divider()

# ================================
# 이탈 확률 분포
# ================================
st.subheader(f"이탈 확률 분포 — {selected_model}")

col_a, col_b = st.columns(2)

with col_a:
    st.bar_chart(
        df["probability"].value_counts(bins=10, sort=False).sort_index()
    )

with col_b:
    dropout_counts = df["predicted"].value_counts().rename({0: "유지", 1: "이탈"})
    st.bar_chart(dropout_counts)

st.divider()

# ================================
# 그룹별 이탈 예측 비율
# ================================
st.subheader("그룹별 이탈 예측 비율")

group_col = st.selectbox("그룹 기준", ["age_band", "gender", "highest_education", "imd_band", "region"])

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

# ================================
# 학생 상세 조회
# ================================
st.subheader("학생 상세 조회")

threshold = st.slider("이탈 확률 임계값 (이상)", 0.0, 1.0, 0.5, 0.05)
high_risk = df[df["probability"] >= threshold].sort_values("probability", ascending=False)

st.write(f"**{len(high_risk):,}명** (확률 ≥ {threshold})")
st.dataframe(
    high_risk[["id_student", "probability", "predicted", "dropout",
               "gender", "age_band", "region", "studied_credits"]].reset_index(drop=True),
    use_container_width=True,
    height=300,
)
