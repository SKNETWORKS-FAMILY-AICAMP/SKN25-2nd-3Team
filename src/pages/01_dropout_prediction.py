import os
import sys
import tempfile

import mlflow
import mlflow.catboost
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sqlalchemy import text

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common_data import get_engine, load_predictions
from models.logistic import EXPERIMENT as LOGISTIC_EXPERIMENT
from models.random_forest import EXPERIMENT as RF_EXPERIMENT, preprocess as rf_preprocess

FEATURE_COLS = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "total_clicks",
    "active_days",
    "unique_resources",
    "num_forum",
    "num_quiz",
    "avg_score",
    "num_assess_attempted",
    "total_weight",
    "module_presentation_length",
]

XGB_FEATURE_COLS = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "module_presentation_length",
    "vle_total_clicks",
    "vle_active_days",
    "vle_unique_resources",
    "vle_avg_clicks_per_day",
    "vle_max_clicks",
    "vle_std_clicks",
]

OFFICIAL_MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost", "TabNet"]

FEATURE_META = {
    "code_module": {
        "label": "수강 모듈",
        "help": "현재 수강 중인 과목(모듈) 코드입니다.",
    },
    "code_presentation": {
        "label": "수강 시기",
        "help": "과정이 개설된 학기/연도 구분입니다.",
    },
    "gender": {
        "label": "성별",
        "help": "학습자 성별입니다.",
    },
    "region": {
        "label": "거주 지역",
        "help": "학습자가 속한 지역입니다.",
    },
    "highest_education": {
        "label": "최종 학력",
        "help": "학습자의 최종 학력 수준입니다.",
    },
    "imd_band": {
        "label": "지역 소득 구간",
        "help": "지역 사회경제 수준 구간(IMD)입니다.",
    },
    "age_band": {
        "label": "연령대",
        "help": "학습자의 연령 구간입니다.",
    },
    "num_of_prev_attempts": {
        "label": "과거 재수강 횟수",
        "help": "같은 과정을 이전에 시도한 횟수입니다.",
    },
    "studied_credits": {
        "label": "수강 학점",
        "help": "현재 수강 중인 총 학점입니다.",
    },
    "disability": {
        "label": "장애 여부",
        "help": "장애 등록 여부입니다.",
    },
    "total_clicks": {
        "label": "학습 페이지 클릭 수",
        "help": "VLE(학습 플랫폼)에서의 전체 클릭 수입니다.",
    },
    "active_days": {
        "label": "활동 일수",
        "help": "학습 플랫폼에 활동한 일수입니다.",
    },
    "unique_resources": {
        "label": "열람한 자료 종류 수",
        "help": "서로 다른 학습 자료를 몇 종류 봤는지입니다.",
    },
    "num_forum": {
        "label": "포럼 활동 수",
        "help": "포럼 글/댓글 등 포럼 관련 활동 수입니다.",
    },
    "num_quiz": {
        "label": "퀴즈 응시 수",
        "help": "퀴즈를 수행한 횟수입니다.",
    },
    "avg_score": {
        "label": "현재 평균 점수",
        "help": "지금까지 평가의 평균 점수입니다.",
    },
    "num_assess_attempted": {
        "label": "제출한 평가 개수",
        "help": "과제/시험 등 제출한 평가 항목 수입니다.",
    },
    "total_weight": {
        "label": "현재까지 반영된 평가 비중(%)",
        "help": "성적에 반영된 평가 비중의 누적값입니다.",
    },
    "module_presentation_length": {
        "label": "과정 전체 기간(일)",
        "help": "해당 과정이 진행되는 전체 일수입니다.",
    },
    "vle_total_clicks": {
        "label": "학습 페이지 클릭 수",
        "help": "과정 초반(컷오프 이전) 학습 플랫폼 전체 클릭 수입니다.",
    },
    "vle_active_days": {
        "label": "활동 일수",
        "help": "과정 초반(컷오프 이전) 학습 활동 일수입니다.",
    },
    "vle_unique_resources": {
        "label": "열람 자료 종류 수",
        "help": "과정 초반(컷오프 이전) 서로 다른 자료를 본 개수입니다.",
    },
    "vle_avg_clicks_per_day": {
        "label": "일평균 클릭 수",
        "help": "과정 초반(컷오프 이전) 일자별 평균 클릭 수입니다.",
    },
    "vle_max_clicks": {
        "label": "일 최대 클릭 수",
        "help": "과정 초반(컷오프 이전) 하루 최대 클릭 수입니다.",
    },
    "vle_std_clicks": {
        "label": "클릭 변동성",
        "help": "과정 초반(컷오프 이전) 일자별 클릭 수의 변동 정도입니다.",
    },
}

VALUE_DISPLAY_MAPS = {
    "gender": {
        "M": "남성",
        "F": "여성",
    },
    "age_band": {
        "0-35": "35세 이하",
        "35-55": "36~55세",
        "55<=": "56세 이상",
    },
    "highest_education": {
        "No Formal Quals": "정규 학력 없음",
        "Lower Than A Level": "A-Level 미만",
        "A Level or Equivalent": "A-Level 또는 동등 학력",
        "HE Qualification": "고등교육 학위",
        "Post Graduate Qualification": "대학원 학위",
    },
    "imd_band": {
        "0-10%": "하위 0~10%",
        "10-20%": "하위 10~20%",
        "20-30%": "하위 20~30%",
        "30-40%": "하위 30~40%",
        "40-50%": "하위 40~50%",
        "50-60%": "하위 50~60%",
        "60-70%": "하위 60~70%",
        "70-80%": "하위 70~80%",
        "80-90%": "하위 80~90%",
        "90-100%": "하위 90~100%",
    },
    "region": {
        "East Anglian Region": "이스트앵글리아",
        "Scotland": "스코틀랜드",
        "South Region": "남부 지역",
        "London Region": "런던",
        "North Western Region": "북서부 지역",
        "South West Region": "남서부 지역",
        "West Midlands Region": "웨스트미들랜즈",
        "East Midlands Region": "이스트미들랜즈",
        "South East Region": "남동부 지역",
        "Wales": "웨일스",
        "Yorkshire Region": "요크셔",
        "North Region": "북부 지역",
        "Ireland": "아일랜드",
    },
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 1.5rem;}
    .hero {
        border-radius: 16px; padding: 18px 20px; margin-bottom: 14px;
        background: linear-gradient(120deg, #1d4ed8 0%, #0ea5e9 55%, #14b8a6 100%);
        color: #ffffff;
    }
    .hero h2 {margin: 0; font-size: 1.35rem;}
    .hero p {margin: 8px 0 0 0; opacity: 0.95;}
    .tip {
        border: 1px solid #e2e8f0; border-radius: 12px; background: #f8fafc;
        padding: 10px 12px; margin-bottom: 10px;
    }
    .section-card {
        border: 1px solid #e2e8f0; border-radius: 14px; background: #ffffff;
        padding: 12px 14px; margin: 10px 0 14px 0;
    }
    .model-pill {
        display: inline-block; border-radius: 999px; padding: 4px 10px;
        font-size: 0.8rem; font-weight: 600; margin-top: 6px;
        border: 1px solid #cbd5e1; background: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h2>학습자 이탈 예측</h2>
      <p>모델별 실시간 예측과 저장된 예측 결과 분석을 한 화면에서 운영할 수 있습니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("현재 UI는 Light 모드 기준으로 가독성과 색상 대비를 최적화했습니다.")


@st.cache_data(ttl=300)
def load_students_for_inference():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM students", engine)


@st.cache_resource
def get_rf_model_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(RF_EXPERIMENT)
    if exp is None:
        raise FileNotFoundError(f"MLflow 실험 '{RF_EXPERIMENT}' 없음")

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    if not runs:
        raise FileNotFoundError("학습된 Random Forest 모델이 없습니다.")

    last_error = None
    for run in runs:
        run_id = run.info.run_id
        if not str(run.info.artifact_uri).startswith("mlflow-artifacts:"):
            continue
        try:
            arts = client.list_artifacts(run_id)
            if not arts:
                continue
        except Exception:
            pass

        try:
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            return model, run_id
        except Exception as e:
            last_error = e

    raise FileNotFoundError(
        "Random Forest 모델 아티팩트를 찾지 못했습니다. "
        "MLflow 서버 아티팩트(run artifact_uri=mlflow-artifacts:/...)가 있는 새 run을 생성하세요. "
        f"(last_error: {last_error})"
    )


@st.cache_resource
def get_logistic_model_mlflow():
    import joblib

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(LOGISTIC_EXPERIMENT)
    if exp is None:
        raise FileNotFoundError(
            f"MLflow 실험 '{LOGISTIC_EXPERIMENT}' 없음. "
            "먼저 `python src/models/logistic.py`를 실행해 실험/모델을 생성하세요."
        )

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    if not runs:
        raise FileNotFoundError(
            "학습된 Logistic 모델이 없습니다. "
            "먼저 `python src/models/logistic.py`를 실행하세요."
        )

    candidate_paths = [
        "model/logistic_model.pkl",
        "logistic_model.pkl",
        "model/model.pkl",
    ]

    last_error = None
    for run in runs:
        run_id = run.info.run_id
        if not str(run.info.artifact_uri).startswith("mlflow-artifacts:"):
            continue

        try:
            top_arts = client.list_artifacts(run_id)
            if not top_arts:
                continue
        except Exception:
            pass

        try:
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            return model, run_id
        except Exception as e:
            last_error = e

        for artifact_path in candidate_paths:
            try:
                with tempfile.TemporaryDirectory() as td:
                    model_file = client.download_artifacts(run_id, artifact_path, td)
                    model = joblib.load(model_file)
                return model, run_id
            except Exception as e:
                last_error = e

    raise FileNotFoundError(
        "Logistic 모델 아티팩트를 찾지 못했습니다. "
        "MLflow 서버 아티팩트(run artifact_uri=mlflow-artifacts:/...)가 있는 새 run을 생성하세요. "
        f"(last_error: {last_error})"
    )


@st.cache_resource
def get_xgboost_model_mlflow():
    from models.xgboost import EXPERIMENT as XGB_EXPERIMENT

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(XGB_EXPERIMENT)
    if exp is None:
        raise FileNotFoundError(
            f"MLflow 실험 '{XGB_EXPERIMENT}' 없음. "
            "먼저 `python src/models/xgboost.py`를 실행해 실험/모델을 생성하세요."
        )

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    if not runs:
        raise FileNotFoundError(
            "학습된 XGBoost 모델이 없습니다. "
            "먼저 `python src/models/xgboost.py`를 실행하세요."
        )

    last_error = None
    for run in runs:
        run_id = run.info.run_id
        if not str(run.info.artifact_uri).startswith("mlflow-artifacts:"):
            continue
        try:
            arts = client.list_artifacts(run_id)
            if not arts:
                continue
        except Exception:
            pass
        try:
            model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
            return model, run_id
        except Exception as e:
            last_error = e

    raise FileNotFoundError(
        "XGBoost 모델 아티팩트를 찾지 못했습니다. "
        "MLflow 서버 아티팩트(run artifact_uri=mlflow-artifacts:/...)가 있는 새 run을 생성하세요. "
        f"(last_error: {last_error})"
    )


@st.cache_resource
def get_catboost_model_mlflow():
    from models.catboost import EXPERIMENT as CAT_EXPERIMENT

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(CAT_EXPERIMENT)
    if exp is None:
        raise FileNotFoundError(
            f"MLflow 실험 '{CAT_EXPERIMENT}' 없음. "
            "먼저 `python src/models/catboost.py`를 실행해 실험/모델을 생성하세요."
        )

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    if not runs:
        raise FileNotFoundError(
            "학습된 CatBoost 모델이 없습니다. "
            "먼저 `python src/models/catboost.py`를 실행하세요."
        )

    last_error = None
    for run in runs:
        run_id = run.info.run_id
        if not str(run.info.artifact_uri).startswith("mlflow-artifacts:"):
            continue
        try:
            arts = client.list_artifacts(run_id)
            if not arts:
                continue
        except Exception:
            pass
        try:
            model = mlflow.catboost.load_model(f"runs:/{run_id}/model")
            return model, run_id
        except Exception as e:
            last_error = e

    raise FileNotFoundError(
        "CatBoost 모델 아티팩트를 찾지 못했습니다. "
        "MLflow 서버 아티팩트(run artifact_uri=mlflow-artifacts:/...)가 있는 새 run을 생성하세요. "
        f"(last_error: {last_error})"
    )


@st.cache_data(ttl=600)
def build_xgboost_inference_data():
    from models import xgboost as xgb_mod

    engine = get_engine()
    df_students = xgb_mod.load_students(engine)
    df_vle = xgb_mod.build_vle_features(engine, xgb_mod.CUTOFF_DAY)
    df_merged = xgb_mod.merge_features(df_students, df_vle)
    X, _, _ = xgb_mod.preprocess(df_merged)
    return df_merged, X


@st.cache_data(ttl=600)
def build_catboost_inference_data():
    from models import catboost as cb_mod

    engine = get_engine()
    df_students = cb_mod.load_students(engine)
    df_vle = cb_mod.build_vle_features(engine, cb_mod.CUTOFF_DAY)
    df_merged = cb_mod.merge_features(df_students, df_vle)
    X, _, _ = cb_mod.preprocess(df_merged)
    return df_merged, X


def build_xgb_defaults(df_ref: pd.DataFrame):
    defaults = {}
    for col in XGB_FEATURE_COLS:
        if pd.api.types.is_numeric_dtype(df_ref[col]):
            defaults[col] = float(df_ref[col].median())
        else:
            mode_vals = df_ref[col].mode(dropna=True)
            defaults[col] = mode_vals.iloc[0] if not mode_vals.empty else df_ref[col].dropna().iloc[0]
    return defaults


def render_xgb_input_form(df_ref: pd.DataFrame, defaults: dict, form_key: str):
    mode = st.radio("입력 모드", ["간소화 입력(추천)", "전체 입력"], horizontal=True, key=f"{form_key}_mode")

    with st.form(form_key):
        if mode == "간소화 입력(추천)":
            st.caption("XGBoost 핵심 피처만 입력합니다. 모듈/개설시기는 기본값으로 자동 입력됩니다.")
            c1, c2 = st.columns(2)
            with c1:
                vle_total_clicks = st.number_input(
                    feature_label("vle_total_clicks"),
                    0.0,
                    200000.0,
                    float(defaults["vle_total_clicks"]),
                    step=50.0,
                    help=feature_help("vle_total_clicks"),
                    key=f"{form_key}_vle_total_clicks_simple",
                )
            with c2:
                vle_active_days = st.number_input(
                    feature_label("vle_active_days"),
                    0.0,
                    100.0,
                    float(defaults["vle_active_days"]),
                    step=1.0,
                    help=feature_help("vle_active_days"),
                    key=f"{form_key}_vle_active_days_simple",
                )
                vle_unique_resources = st.number_input(
                    feature_label("vle_unique_resources"),
                    0.0,
                    1000.0,
                    float(defaults["vle_unique_resources"]),
                    step=1.0,
                    help=feature_help("vle_unique_resources"),
                    key=f"{form_key}_vle_unique_resources_simple",
                )

            row_dict = defaults.copy()
            row_dict.update(
                {
                    "vle_total_clicks": vle_total_clicks,
                    "vle_active_days": vle_active_days,
                    "vle_unique_resources": vle_unique_resources,
                }
            )
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                row_dict = {
                    "gender": st.selectbox(
                        feature_label("gender"),
                        ["M", "F"],
                        help=feature_help("gender"),
                        key=f"{form_key}_gender",
                    ),
                    "region": st.selectbox(
                        feature_label("region"),
                        sorted(df_ref["region"].dropna().unique().tolist()),
                        help=feature_help("region"),
                        key=f"{form_key}_region",
                    ),
                    "highest_education": st.selectbox(
                        feature_label("highest_education"),
                        sorted(df_ref["highest_education"].dropna().unique().tolist()),
                        help=feature_help("highest_education"),
                        key=f"{form_key}_highest_education",
                    ),
                    "imd_band": st.selectbox(
                        feature_label("imd_band"),
                        sorted(df_ref["imd_band"].dropna().unique().tolist()),
                        help=feature_help("imd_band"),
                        key=f"{form_key}_imd_band",
                    ),
                    "age_band": st.selectbox(
                        feature_label("age_band"),
                        sorted(df_ref["age_band"].dropna().unique().tolist()),
                        help=feature_help("age_band"),
                        key=f"{form_key}_age_band",
                    ),
                }
            with c2:
                row_dict.update(
                    {
                        "num_of_prev_attempts": st.number_input(
                            feature_label("num_of_prev_attempts"),
                            0,
                            10,
                            int(defaults["num_of_prev_attempts"]),
                            help=feature_help("num_of_prev_attempts"),
                            key=f"{form_key}_num_prev",
                        ),
                        "studied_credits": st.number_input(
                            feature_label("studied_credits"),
                            0,
                            600,
                            int(defaults["studied_credits"]),
                            step=30,
                            help=feature_help("studied_credits"),
                            key=f"{form_key}_credits",
                        ),
                        "disability": st.selectbox(
                            feature_label("disability"),
                            ["N", "Y"],
                            help=feature_help("disability"),
                            key=f"{form_key}_disability",
                        ),
                        "module_presentation_length": st.number_input(
                            feature_label("module_presentation_length"),
                            0,
                            400,
                            int(defaults["module_presentation_length"]),
                            help=feature_help("module_presentation_length"),
                            key=f"{form_key}_module_len",
                        ),
                    }
                )
            with c3:
                row_dict.update(
                    {
                        "vle_total_clicks": st.number_input(
                            feature_label("vle_total_clicks"),
                            0.0,
                            200000.0,
                            float(defaults["vle_total_clicks"]),
                            step=50.0,
                            help=feature_help("vle_total_clicks"),
                            key=f"{form_key}_vle_total_clicks",
                        ),
                        "vle_active_days": st.number_input(
                            feature_label("vle_active_days"),
                            0.0,
                            100.0,
                            float(defaults["vle_active_days"]),
                            step=1.0,
                            help=feature_help("vle_active_days"),
                            key=f"{form_key}_vle_active_days",
                        ),
                        "vle_unique_resources": st.number_input(
                            feature_label("vle_unique_resources"),
                            0.0,
                            1000.0,
                            float(defaults["vle_unique_resources"]),
                            step=1.0,
                            help=feature_help("vle_unique_resources"),
                            key=f"{form_key}_vle_unique_resources",
                        ),
                        "vle_avg_clicks_per_day": st.number_input(
                            feature_label("vle_avg_clicks_per_day"),
                            0.0,
                            10000.0,
                            float(defaults["vle_avg_clicks_per_day"]),
                            step=10.0,
                            help=feature_help("vle_avg_clicks_per_day"),
                            key=f"{form_key}_vle_avg_clicks_per_day",
                        ),
                        "vle_max_clicks": st.number_input(
                            feature_label("vle_max_clicks"),
                            0.0,
                            10000.0,
                            float(defaults["vle_max_clicks"]),
                            step=10.0,
                            help=feature_help("vle_max_clicks"),
                            key=f"{form_key}_vle_max_clicks",
                        ),
                        "vle_std_clicks": st.number_input(
                            feature_label("vle_std_clicks"),
                            0.0,
                            5000.0,
                            float(defaults["vle_std_clicks"]),
                            step=1.0,
                            help=feature_help("vle_std_clicks"),
                            key=f"{form_key}_vle_std_clicks",
                        ),
                    }
                )

            with st.expander("고급 옵션: 모듈/개설시기 변경", expanded=False):
                row_dict["code_module"] = st.selectbox(
                    feature_label("code_module"),
                    sorted(df_ref["code_module"].dropna().unique().tolist()),
                    index=sorted(df_ref["code_module"].dropna().unique().tolist()).index(defaults["code_module"])
                    if defaults["code_module"] in sorted(df_ref["code_module"].dropna().unique().tolist())
                    else 0,
                    help=feature_help("code_module"),
                    key=f"{form_key}_code_module_adv",
                )
                row_dict["code_presentation"] = st.selectbox(
                    feature_label("code_presentation"),
                    sorted(df_ref["code_presentation"].dropna().unique().tolist()),
                    index=sorted(df_ref["code_presentation"].dropna().unique().tolist()).index(defaults["code_presentation"])
                    if defaults["code_presentation"] in sorted(df_ref["code_presentation"].dropna().unique().tolist())
                    else 0,
                    help=feature_help("code_presentation"),
                    key=f"{form_key}_code_presentation_adv",
                )

            for col in XGB_FEATURE_COLS:
                row_dict.setdefault(col, defaults[col])

        st.markdown("#### 저장 옵션")
        save_prediction = st.checkbox("예측 결과 저장", value=False, key=f"{form_key}_save")
        user_id_text = st.text_input("학생 ID (선택, 숫자만)", value="", key=f"{form_key}_student_id")
        submitted = st.form_submit_button("예측 실행", width="stretch")

    return submitted, row_dict, save_prediction, user_id_text


def feature_label(key: str) -> str:
    return FEATURE_META[key]["label"]


def feature_help(key: str) -> str:
    return FEATURE_META[key]["help"]


def display_group_value(col: str, value):
    mapping = VALUE_DISPLAY_MAPS.get(col, {})
    return mapping.get(value, value)


def normalize_model_name(model_name: str) -> str:
    name = str(model_name).lower().strip()
    if "logistic" in name:
        return "Logistic Regression"
    if "random_forest" in name:
        return "Random Forest"
    if "catboost" in name:
        return "CatBoost"
    if "tabnet" in name:
        return "TabNet"
    if "xgboost" in name:
        return "XGBoost"
    return str(model_name)


def model_color(model_name: str) -> str:
    if model_name == "Random Forest":
        return "#059669"
    if model_name == "Logistic Regression":
        return "#2563eb"
    if model_name == "CatBoost":
        return "#0f766e"
    if model_name == "XGBoost":
        return "#ea580c"
    if model_name == "TabNet":
        return "#7c3aed"
    return "#334155"


def save_live_prediction(model_name, id_student, pred, prob, run_id):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS live_predictions (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    id_student INT NULL,
                    model_name VARCHAR(50),
                    predicted TINYINT,
                    probability FLOAT,
                    run_id VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )

    df_save = pd.DataFrame(
        [
            {
                "id_student": id_student,
                "model_name": model_name,
                "predicted": int(pred),
                "probability": float(prob),
                "run_id": str(run_id),
            }
        ]
    )
    df_save.to_sql("live_predictions", engine, if_exists="append", index=False)


def build_defaults(df_ref: pd.DataFrame):
    defaults = {}
    for col in FEATURE_COLS:
        if pd.api.types.is_numeric_dtype(df_ref[col]):
            defaults[col] = float(df_ref[col].median())
        else:
            mode_vals = df_ref[col].mode(dropna=True)
            defaults[col] = mode_vals.iloc[0] if not mode_vals.empty else df_ref[col].dropna().iloc[0]
    return defaults


def render_input_form(df_ref: pd.DataFrame, defaults: dict, form_key: str):
    mode = st.radio("입력 모드", ["간소화 입력(추천)", "전체 입력"], horizontal=True, key=f"{form_key}_mode")

    with st.form(form_key):
        if mode == "간소화 입력(추천)":
            st.caption("중요 피처 5개만 입력합니다. 나머지는 기본값으로 자동 입력됩니다.")
            c1, c2 = st.columns(2)
            with c1:
                total_weight = st.number_input(
                    feature_label("total_weight"),
                    0.0,
                    500.0,
                    float(defaults["total_weight"]),
                    step=1.0,
                    help=feature_help("total_weight"),
                )
                num_assess_attempted = st.number_input(
                    feature_label("num_assess_attempted"),
                    0.0,
                    50.0,
                    float(defaults["num_assess_attempted"]),
                    step=1.0,
                    help=feature_help("num_assess_attempted"),
                )
                avg_score = st.number_input(
                    feature_label("avg_score"),
                    0.0,
                    100.0,
                    float(defaults["avg_score"]),
                    step=0.1,
                    help=feature_help("avg_score"),
                )
            with c2:
                total_clicks = st.number_input(
                    feature_label("total_clicks"),
                    0.0,
                    50000.0,
                    float(defaults["total_clicks"]),
                    step=100.0,
                    help=feature_help("total_clicks"),
                )
                unique_resources = st.number_input(
                    feature_label("unique_resources"),
                    0.0,
                    1000.0,
                    float(defaults["unique_resources"]),
                    step=1.0,
                    help=feature_help("unique_resources"),
                )

            row_dict = defaults.copy()
            row_dict.update(
                {
                    "total_weight": total_weight,
                    "num_assess_attempted": num_assess_attempted,
                    "avg_score": avg_score,
                    "total_clicks": total_clicks,
                    "unique_resources": unique_resources,
                }
            )

        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                row_dict = {
                    "code_module": st.selectbox(
                        feature_label("code_module"),
                        sorted(df_ref["code_module"].dropna().unique().tolist()),
                        key=f"{form_key}_code_module",
                        help=feature_help("code_module"),
                    ),
                    "code_presentation": st.selectbox(
                        feature_label("code_presentation"),
                        sorted(df_ref["code_presentation"].dropna().unique().tolist()),
                        key=f"{form_key}_code_presentation",
                        help=feature_help("code_presentation"),
                    ),
                    "gender": st.selectbox(
                        feature_label("gender"),
                        ["M", "F"],
                        key=f"{form_key}_gender",
                        help=feature_help("gender"),
                    ),
                    "region": st.selectbox(
                        feature_label("region"),
                        sorted(df_ref["region"].dropna().unique().tolist()),
                        key=f"{form_key}_region",
                        help=feature_help("region"),
                    ),
                    "highest_education": st.selectbox(
                        feature_label("highest_education"),
                        sorted(df_ref["highest_education"].dropna().unique().tolist()),
                        key=f"{form_key}_highest_education",
                        help=feature_help("highest_education"),
                    ),
                    "imd_band": st.selectbox(
                        feature_label("imd_band"),
                        sorted(df_ref["imd_band"].dropna().unique().tolist()),
                        key=f"{form_key}_imd_band",
                        help=feature_help("imd_band"),
                    ),
                    "age_band": st.selectbox(
                        feature_label("age_band"),
                        sorted(df_ref["age_band"].dropna().unique().tolist()),
                        key=f"{form_key}_age_band",
                        help=feature_help("age_band"),
                    ),
                }
            with c2:
                row_dict.update(
                    {
                        "num_of_prev_attempts": st.number_input(
                            feature_label("num_of_prev_attempts"),
                            0,
                            10,
                            int(defaults["num_of_prev_attempts"]),
                            key=f"{form_key}_num_prev",
                            help=feature_help("num_of_prev_attempts"),
                        ),
                        "studied_credits": st.number_input(
                            feature_label("studied_credits"),
                            0,
                            600,
                            int(defaults["studied_credits"]),
                            step=30,
                            key=f"{form_key}_credits",
                            help=feature_help("studied_credits"),
                        ),
                        "disability": st.selectbox(
                            feature_label("disability"),
                            ["N", "Y"],
                            key=f"{form_key}_disability",
                            help=feature_help("disability"),
                        ),
                        "total_clicks": st.number_input(
                            feature_label("total_clicks"),
                            0.0,
                            50000.0,
                            float(defaults["total_clicks"]),
                            step=100.0,
                            key=f"{form_key}_total_clicks",
                            help=feature_help("total_clicks"),
                        ),
                        "active_days": st.number_input(
                            feature_label("active_days"),
                            0.0,
                            365.0,
                            float(defaults["active_days"]),
                            step=1.0,
                            key=f"{form_key}_active_days",
                            help=feature_help("active_days"),
                        ),
                        "unique_resources": st.number_input(
                            feature_label("unique_resources"),
                            0.0,
                            1000.0,
                            float(defaults["unique_resources"]),
                            step=1.0,
                            key=f"{form_key}_unique_resources",
                            help=feature_help("unique_resources"),
                        ),
                    }
                )
            with c3:
                row_dict.update(
                    {
                        "num_forum": st.number_input(
                            feature_label("num_forum"),
                            0.0,
                            500.0,
                            float(defaults["num_forum"]),
                            step=1.0,
                            key=f"{form_key}_num_forum",
                            help=feature_help("num_forum"),
                        ),
                        "num_quiz": st.number_input(
                            feature_label("num_quiz"),
                            0.0,
                            100.0,
                            float(defaults["num_quiz"]),
                            step=1.0,
                            key=f"{form_key}_num_quiz",
                            help=feature_help("num_quiz"),
                        ),
                        "avg_score": st.number_input(
                            feature_label("avg_score"),
                            0.0,
                            100.0,
                            float(defaults["avg_score"]),
                            step=0.1,
                            key=f"{form_key}_avg_score",
                            help=feature_help("avg_score"),
                        ),
                        "num_assess_attempted": st.number_input(
                            feature_label("num_assess_attempted"),
                            0.0,
                            50.0,
                            float(defaults["num_assess_attempted"]),
                            step=1.0,
                            key=f"{form_key}_num_assess",
                            help=feature_help("num_assess_attempted"),
                        ),
                        "total_weight": st.number_input(
                            feature_label("total_weight"),
                            0.0,
                            500.0,
                            float(defaults["total_weight"]),
                            step=1.0,
                            key=f"{form_key}_total_weight",
                            help=feature_help("total_weight"),
                        ),
                        "module_presentation_length": st.number_input(
                            feature_label("module_presentation_length"),
                            0,
                            400,
                            int(defaults["module_presentation_length"]),
                            key=f"{form_key}_module_len",
                            help=feature_help("module_presentation_length"),
                        ),
                    }
                )

            for col in FEATURE_COLS:
                row_dict.setdefault(col, defaults[col])

        st.markdown("#### 저장 옵션")
        save_prediction = st.checkbox("예측 결과 저장", value=False, key=f"{form_key}_save")
        user_id_text = st.text_input("학생 ID (선택, 숫자만)", value="", key=f"{form_key}_student_id")
        submitted = st.form_submit_button("예측 실행", width="stretch")

    return submitted, row_dict, save_prediction, user_id_text


def rf_inference_ui(df_ref: pd.DataFrame):
    rf_model, rf_run_id = get_rf_model_mlflow()
    st.caption(f"Random Forest 최신 모델 실험 ID: `{rf_run_id}`")

    defaults = build_defaults(df_ref)
    submitted, row_dict, save_prediction, user_id_text = render_input_form(df_ref, defaults, "rf_form")

    if submitted:
        row_dict["dropout"] = 0
        input_row = pd.DataFrame([row_dict])

        df_combined = pd.concat([df_ref, input_row], ignore_index=True)
        df_enc = rf_preprocess(df_combined)
        X_input = df_enc.drop(columns=["dropout"]).iloc[[-1]].values

        prob = float(rf_model.predict_proba(X_input)[0, 1])
        pred = int(rf_model.predict(X_input)[0])

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("이탈 확률", f"{prob:.1%}")
        col2.metric("예측 결과", "이탈" if pred == 1 else "유지")

        if save_prediction:
            id_student = None
            if user_id_text.strip():
                if not user_id_text.strip().isdigit():
                    st.error("학생 ID는 숫자만 입력 가능합니다.")
                    st.stop()
                id_student = int(user_id_text.strip())
            save_live_prediction("Random Forest", id_student, pred, prob, rf_run_id)
            st.success("예측 결과를 저장했습니다.")


def logistic_inference_ui(df_ref: pd.DataFrame):
    model, run_id = get_logistic_model_mlflow()
    st.caption(f"Logistic Regression 최신 모델 실험 ID: `{run_id}`")

    defaults = build_defaults(df_ref)
    submitted, row_dict, save_prediction, user_id_text = render_input_form(df_ref, defaults, "logistic_form")

    if submitted:
        X_input = pd.DataFrame([{k: row_dict[k] for k in FEATURE_COLS}])

        prob = float(model.predict_proba(X_input)[0, 1])
        pred = int(model.predict(X_input)[0])

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("이탈 확률", f"{prob:.1%}")
        col2.metric("예측 결과", "이탈" if pred == 1 else "유지")

        if save_prediction:
            id_student = None
            if user_id_text.strip():
                if not user_id_text.strip().isdigit():
                    st.error("학생 ID는 숫자만 입력 가능합니다.")
                    st.stop()
                id_student = int(user_id_text.strip())
            save_live_prediction("Logistic Regression", id_student, pred, prob, run_id)
            st.success("예측 결과를 저장했습니다.")


def tabnet_inference_ui(df_ref: pd.DataFrame):
    from models.tabnet import load_model, predict, preprocess

    @st.cache_resource
    def get_tabnet_model():
        return load_model()

    model = get_tabnet_model()
    df_processed = preprocess(df_ref.copy())
    X_all = df_processed.drop(columns=["dropout"]).values

    student_ids = df_ref["id_student"].tolist()
    selected_id = st.selectbox("학생 ID", student_ids)

    if st.button("TabNet 예측 실행"):
        idx = df_ref[df_ref["id_student"] == selected_id].index[0]
        pos = df_ref.index.get_loc(idx)
        X_input = X_all[pos : pos + 1]

        predicted, probability = predict(model, X_input)
        pred = int(predicted[0])
        prob = float(probability[0])
        actual = int(df_ref.loc[idx, "dropout"])

        col1, col2, col3 = st.columns(3)
        col1.metric("이탈 확률", f"{prob:.1%}")
        col2.metric("예측 결과", "이탈" if pred == 1 else "유지")
        col3.metric("실제 결과", "이탈" if actual == 1 else "유지")


def xgboost_inference_ui():
    from models import xgboost as xgb_mod

    model, run_id = get_xgboost_model_mlflow()
    st.caption(f"XGBoost 최신 모델 실험 ID: `{run_id}`")

    df_xgb, _ = build_xgboost_inference_data()
    if df_xgb.empty:
        st.warning("XGBoost 추론용 데이터가 없습니다.")
        return

    defaults = build_xgb_defaults(df_xgb)
    submitted, row_dict, save_prediction, user_id_text = render_xgb_input_form(df_xgb, defaults, "xgb_form")

    if submitted:
        input_row = pd.DataFrame([{"id_student": -1, **row_dict, "dropout": 0}])
        df_combined = pd.concat([df_xgb, input_row], ignore_index=True)
        X_enc, _, _ = xgb_mod.preprocess(df_combined)
        X_input = X_enc.iloc[[-1]]

        prob = float(model.predict_proba(X_input)[0, 1])
        pred = int(model.predict(X_input)[0])

        c1, c2 = st.columns(2)
        c1.metric("이탈 확률", f"{prob:.1%}")
        c2.metric("예측 결과", "이탈" if pred == 1 else "유지")

        if save_prediction:
            id_student = None
            if user_id_text.strip():
                if not user_id_text.strip().isdigit():
                    st.error("학생 ID는 숫자만 입력 가능합니다.")
                    st.stop()
                id_student = int(user_id_text.strip())
            save_live_prediction("XGBoost", id_student, pred, prob, run_id)
            st.success("예측 결과를 저장했습니다.")


def catboost_inference_ui():
    from models import catboost as cb_mod

    model, run_id = get_catboost_model_mlflow()
    st.caption(f"CatBoost 최신 모델 실험 ID: `{run_id}`")

    df_cb, _ = build_catboost_inference_data()
    if df_cb.empty:
        st.warning("CatBoost 추론용 데이터가 없습니다.")
        return

    defaults = build_xgb_defaults(df_cb)
    submitted, row_dict, save_prediction, user_id_text = render_xgb_input_form(df_cb, defaults, "catboost_form")

    if submitted:
        input_row = pd.DataFrame([{"id_student": -1, **row_dict, "dropout": 0}])
        df_combined = pd.concat([df_cb, input_row], ignore_index=True)
        X_enc, _, _ = cb_mod.preprocess(df_combined)
        X_input = X_enc.iloc[[-1]]

        prob = float(model.predict_proba(X_input)[0, 1])
        pred = int(model.predict(X_input)[0])

        c1, c2 = st.columns(2)
        c1.metric("이탈 확률", f"{prob:.1%}")
        c2.metric("예측 결과", "이탈" if pred == 1 else "유지")

        if save_prediction:
            id_student = None
            if user_id_text.strip():
                if not user_id_text.strip().isdigit():
                    st.error("학생 ID는 숫자만 입력 가능합니다.")
                    st.stop()
                id_student = int(user_id_text.strip())
            save_live_prediction("CatBoost", id_student, pred, prob, run_id)
            st.success("예측 결과를 저장했습니다.")


tab_infer, tab_analysis = st.tabs(["실시간 예측", "예측 결과 분석"])

with tab_infer:
    st.markdown(
        """
        <div class="tip">
          간소화 입력은 핵심 피처만 직접 입력하고, 나머지는 데이터 중앙값/최빈값으로 자동 보정합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        df_students = load_students_for_inference()
    except Exception as e:
        st.error(f"DB 연결 실패: {e}")
        st.stop()

    model_choice = st.selectbox(
        "모델 선택",
        ["Random Forest", "Logistic Regression", "XGBoost", "CatBoost"],
    )
    chip_color = model_color(model_choice)
    st.markdown(
        f"<span class='model-pill' style='color:{chip_color}; border-color:{chip_color}33; background:{chip_color}14;'>선택 모델: {model_choice}</span>",
        unsafe_allow_html=True,
    )
    st.caption("TabNet은 PyTorch 환경 의존성으로 인해 여기서는 제외되며, `예측 결과 분석` 탭에서 결과를 확인할 수 있습니다.")

    if model_choice == "Random Forest":
        try:
            rf_inference_ui(df_students)
        except Exception as e:
            st.error(f"Random Forest 추론 실패: {e}")

    elif model_choice == "Logistic Regression":
        try:
            logistic_inference_ui(df_students)
        except Exception as e:
            st.error(f"Logistic Regression 추론 실패: {e}")

    elif model_choice == "XGBoost":
        try:
            xgboost_inference_ui()
        except Exception as e:
            st.error(f"XGBoost 추론 실패: {e}")

    else:
        try:
            catboost_inference_ui()
        except Exception as e:
            st.error(f"CatBoost 추론 실패: {e}")

with tab_analysis:
    st.markdown(
        """
        <div class="tip">
          모델별 예측 성능, 확률 분포, 그룹별 위험 비율을 비교해 운영 우선순위를 잡을 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        df_pred = load_predictions()
    except Exception as e:
        st.error(f"DB 연결 실패: {e}")
        st.stop()

    if df_pred.empty:
        st.warning("예측 결과 테이블(`predictions`)에 데이터가 없습니다.")
        st.stop()

    df_pred = df_pred.copy()
    df_pred["display_model"] = df_pred["model_name"].apply(normalize_model_name)

    models = OFFICIAL_MODELS.copy()
    selected_model = st.selectbox("분석할 모델 선택", models)
    df = df_pred[df_pred["display_model"] == selected_model].copy()

    total = len(df)
    if total == 0:
        st.warning(f"`{selected_model}` 데이터가 아직 없습니다. 해당 모델 학습/저장을 먼저 실행하세요.")
        st.stop()

    predicted_dropout = int(df["predicted"].sum())
    actual_dropout = int(df["dropout"].sum()) if "dropout" in df.columns else None
    avg_prob = df["probability"].mean()

    selected_color = model_color(selected_model)
    st.markdown(
        f"<span class='model-pill' style='color:{selected_color}; border-color:{selected_color}33; background:{selected_color}14;'>현재 분석 모델: {selected_model}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 학생 수", f"{total:,}")
    c2.metric("예측 이탈", f"{predicted_dropout:,}", f"{predicted_dropout / total * 100:.1f}%")
    if actual_dropout is not None:
        c3.metric("실제 이탈", f"{actual_dropout:,}", f"{actual_dropout / total * 100:.1f}%")
    c4.metric("평균 이탈 확률", f"{avg_prob:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

    if len(models) > 1:
        rows = []
        for m in models:
            d = df_pred[df_pred["display_model"] == m]
            if "dropout" in d.columns and d["dropout"].notna().any():
                y_true = d["dropout"].dropna().astype(int)
                y_pred = d.loc[y_true.index, "predicted"].astype(int)
                y_prob = d.loc[y_true.index, "probability"]
                rows.append(
                    {
                        "모델": m,
                        "F1": f"{f1_score(y_true, y_pred):.4f}",
                        "AUC": f"{roc_auc_score(y_true, y_prob):.4f}",
                        "정확도": f"{accuracy_score(y_true, y_pred):.4f}",
                    }
                )
        if rows:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("모델별 성능 비교")
            score_df = pd.DataFrame(rows).sort_values("AUC", ascending=False).set_index("모델")
            st.dataframe(score_df, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("예측 결과 시각화")
    ca, cb = st.columns(2)
    with ca:
        fig_prob = px.histogram(
            df,
            x="probability",
            nbins=20,
            color_discrete_sequence=[selected_color],
            labels={"probability": "이탈 확률", "count": "학생 수"},
            title="이탈 확률 분포",
        )
        fig_prob.update_layout(xaxis_title="이탈 확률", yaxis_title="학생 수")
        st.plotly_chart(fig_prob, width="stretch")
    with cb:
        pred_counts = (
            df["predicted"]
            .value_counts()
            .rename(index={0: "유지 예측", 1: "이탈 예측"})
            .reset_index()
        )
        pred_counts.columns = ["예측 결과", "학생 수"]
        fig_pred = px.bar(
            pred_counts,
            x="예측 결과",
            y="학생 수",
            text="학생 수",
            color="예측 결과",
            color_discrete_map={"유지 예측": "#0ea5e9", "이탈 예측": "#ef4444"},
            labels={"예측 결과": "예측 결과", "학생 수": "학생 수"},
            title="예측 결과 분포",
        )
        fig_pred.update_traces(textposition="outside")
        st.plotly_chart(fig_pred, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("그룹별 이탈 예측 비율")
    group_options = {
        "연령대": "age_band",
        "성별": "gender",
        "최종 학력": "highest_education",
        "지역 소득 구간": "imd_band",
        "거주 지역": "region",
    }
    selected_group_label = st.selectbox("비교 기준", list(group_options.keys()))
    group_col = group_options[selected_group_label]
    group_label_map = {
        "age_band": "연령대",
        "gender": "성별",
        "highest_education": "최종 학력",
        "imd_band": "지역 소득 구간",
        "region": "거주 지역",
    }

    if group_col in df.columns:
        display_col = group_label_map[group_col]
        group_df = (
            df.groupby(group_col)["predicted"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "이탈 예측 수", "count": "전체 학생 수"})
            .reset_index()
        )
        group_df[group_col] = group_df[group_col].apply(lambda v: display_group_value(group_col, v))
        group_df = group_df.rename(columns={group_col: display_col})
        group_df["이탈 예측 비율(%)"] = (group_df["이탈 예측 수"] / group_df["전체 학생 수"] * 100).round(1)
        group_df = group_df.sort_values("이탈 예측 비율(%)", ascending=False)

        fig_group = px.bar(
            group_df,
            x=display_col,
            y="이탈 예측 비율(%)",
            text="이탈 예측 비율(%)",
            color="이탈 예측 비율(%)",
            color_continuous_scale=["#bfdbfe", "#3b82f6", "#1e3a8a"],
            labels={display_col: display_col, "이탈 예측 비율(%)": "이탈 예측 비율(%)"},
            title=f"{display_col}별 이탈 예측 비율",
        )
        fig_group.update_traces(textposition="outside")
        st.plotly_chart(fig_group, width="stretch")
        st.dataframe(group_df, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
