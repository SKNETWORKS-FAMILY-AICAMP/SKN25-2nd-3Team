"""
CatBoost 이탈 예측 모델
- train(engine) : 학습 + MLflow 로깅 + predictions 저장
- load_model()  : 저장된 모델 로드 (Streamlit용)
- predict()     : 추론 (Streamlit용)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sqlalchemy import create_engine, text

# `python src/models/catboost.py` 직접 실행 시, 파일명이 라이브러리명과 같아
# import shadowing이 발생할 수 있어 현재 디렉토리를 잠시 sys.path에서 제거한다.
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_RESTORE_PATH = False
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
    _RESTORE_PATH = True
from catboost import CatBoostClassifier
if _RESTORE_PATH:
    sys.path.insert(0, _THIS_DIR)

load_dotenv()

# ================================
# 상수
# ================================
EXPERIMENT = "catboost_stu_vle_v2"
MODEL_NAME = "catboost_dropout"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "catboost_model.cbm")
RANDOM_STATE = 42
TEST_SIZE = 0.2
CUTOFF_DAY = 60

IMD_ORDER = [
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
]
AGE_ORDER = ["0-35", "35-55", "55<="]
EDU_ORDER = [
    "No Formal Quals",
    "Lower Than A Level",
    "A Level or Equivalent",
    "HE Qualification",
    "Post Graduate Qualification",
]

LEAKAGE_COLS = [
    "total_clicks",
    "active_days",
    "unique_resources",
    "num_forum",
    "num_quiz",
    "avg_score",
    "num_assess_attempted",
    "total_weight",
]

PARAM_DIST = {
    "iterations": [500, 700, 1000],
    "depth": [4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "l2_leaf_reg": [1, 3, 5, 10],
    "bagging_temperature": [0, 0.5, 1.0],
    "border_count": [32, 64, 128],
}


def get_engine():
    return create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


def load_students(engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM students", engine)
    print(f"students : {df.shape}")
    return df


def build_vle_features(engine, cutoff_day: int) -> pd.DataFrame:
    query = f"""
        SELECT
            id_student,
            code_module,
            code_presentation,
            SUM(sum_click) AS vle_total_clicks,
            COUNT(DISTINCT date) AS vle_active_days,
            COUNT(DISTINCT id_site) AS vle_unique_resources,
            AVG(sum_click) AS vle_avg_clicks_per_day,
            MAX(sum_click) AS vle_max_clicks,
            COALESCE(STDDEV_POP(sum_click), 0) AS vle_std_clicks
        FROM studentVle
        WHERE date <= {int(cutoff_day)}
        GROUP BY id_student, code_module, code_presentation
    """
    df = pd.read_sql(query, engine)
    print(f"vle_features : {df.shape}")
    return df


def merge_features(df_students: pd.DataFrame, df_vle_feat: pd.DataFrame) -> pd.DataFrame:
    df_base = df_students.drop(columns=LEAKAGE_COLS, errors="ignore")
    df = df_base.merge(df_vle_feat, on=["id_student", "code_module", "code_presentation"], how="left")

    for col in [
        "vle_total_clicks",
        "vle_active_days",
        "vle_unique_resources",
        "vle_avg_clicks_per_day",
        "vle_max_clicks",
        "vle_std_clicks",
    ]:
        df[col] = df[col].fillna(0)

    print(f"merged : {df.shape}")
    return df


def preprocess(df: pd.DataFrame):
    df = df.copy()
    id_arr = df["id_student"].values

    y = df["dropout"].astype(int)
    X = df.drop(columns=["id_student", "dropout"], errors="ignore")

    X["gender"] = X["gender"].map({"M": 1, "F": 0})
    X["disability"] = X["disability"].map({"Y": 1, "N": 0})

    oe = OrdinalEncoder(
        categories=[IMD_ORDER, AGE_ORDER, EDU_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    X[["imd_band", "age_band", "highest_education"]] = oe.fit_transform(
        X[["imd_band", "age_band", "highest_education"]]
    )

    for col in ["region", "code_module", "code_presentation"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return X, y, id_arr


def train(engine=None, cutoff_day: int = CUTOFF_DAY):
    if engine is None:
        engine = get_engine()

    df_students = load_students(engine)
    df_vle = build_vle_features(engine, cutoff_day)
    df = merge_features(df_students, df_vle)

    X, y, id_arr = preprocess(df)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        X.index,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    search = RandomizedSearchCV(
        estimator=CatBoostClassifier(
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
        param_distributions=PARAM_DIST,
        n_iter=20,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=1,
    )
    search.fit(X_train, y_train)

    model = search.best_estimator_
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
    }
    print("\n".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "catboost")
    os.makedirs(out_dir, exist_ok=True)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="catboost") as run:
        mlflow.log_params({**search.best_params_, "cutoff_day": cutoff_day})
        mlflow.log_metrics(metrics)

        mlflow.catboost.log_model(model, artifact_path="model")

        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Stay", "Dropout"]).plot(ax=ax)
        ax.set_title(f"Confusion Matrix - CatBoost (cutoff={cutoff_day})")
        cm_path = os.path.join(out_dir, "confusion_matrix.png")
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path)

        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer(X_test)
            shap.plots.beeswarm(shap_vals, max_display=20, show=False)
            shap_path = os.path.join(out_dir, "shap_beeswarm.png")
            plt.tight_layout()
            plt.savefig(shap_path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(shap_path)
        except Exception as e:
            print(f"SHAP 생성 생략: {e}")

        run_id = run.info.run_id

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)

    pred_df = pd.DataFrame(
        {
            "id_student": id_arr[idx_test],
            "model_name": MODEL_NAME,
            "predicted": y_pred,
            "probability": y_prob,
            "run_id": run_id,
        }
    )

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM predictions WHERE model_name = :name"), {"name": MODEL_NAME})
        conn.commit()

    pred_df.to_sql("predictions", engine, if_exists="append", index=False)
    print(f"predictions 저장 완료: {len(pred_df)}건")
    print(f"MLflow run_id: {run_id}")

    return run_id


# ================================
# 추론 (Streamlit용)
# ================================
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


def predict(model, X: np.ndarray):
    predicted = model.predict(X).astype(int)
    probability = model.predict_proba(X)[:, 1]
    return predicted, probability


if __name__ == "__main__":
    train()
