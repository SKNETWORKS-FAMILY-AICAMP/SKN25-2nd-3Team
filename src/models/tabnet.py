"""
TabNet 이탈 예측 모델
- train(engine) : 학습 + MLflow 로깅 + predictions 저장
- predict(X)    : 저장된 모델로 추론 (Streamlit용)
"""
import os
import numpy as np
import pandas as pd
import torch
import mlflow
from dotenv import load_dotenv
from sqlalchemy import text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, recall_score, roc_auc_score,
)
from pytorch_tabnet.tab_model import TabNetClassifier

load_dotenv()

# ================================
# 상수
# ================================
IMD_BAND_ORDER = [
    "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
]
AGE_BAND_ORDER = ["0-35", "35-55", "55<="]
EDUCATION_ORDER = [
    "No Formal Quals",
    "Lower Than A Level",
    "A Level or Equivalent",
    "HE Qualification",
    "Post Graduate Qualification",
]
CAT_COLS = ["region", "code_module", "code_presentation",
            "highest_education", "imd_band", "age_band"]

MODEL_PATH     = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'tabnet_model')
MLFLOW_EXPERIMENT = "tabnet-dropout-prediction"
RUN_NAME       = "tabnet"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

PARAMS = {
    "n_d": 16, "n_a": 16, "n_steps": 5, "gamma": 1.3,
    "lr": 2e-2, "batch_size": 1024, "virtual_batch_size": 128,
    "max_epochs": 100, "patience": 15,
}


# ================================
# 전처리
# ================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["id_student"])
    df["gender"]     = df["gender"].map({"M": 1, "F": 0})
    df["disability"] = df["disability"].map({"Y": 1, "N": 0})
    df["imd_band"]   = df["imd_band"].fillna("40-50%")

    oe = OrdinalEncoder(
        categories=[IMD_BAND_ORDER, AGE_BAND_ORDER, EDUCATION_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    df[["imd_band", "age_band", "highest_education"]] = oe.fit_transform(
        df[["imd_band", "age_band", "highest_education"]]
    ).astype(int)

    le = LabelEncoder()
    for col in ["region", "code_module", "code_presentation"]:
        df[col] = le.fit_transform(df[col])

    return df


# ================================
# 학습
# ================================
def train(engine):
    df = pd.read_sql("SELECT * FROM students", engine)

    df_clean = preprocess(df.copy())
    X = df_clean.drop(columns=["dropout"])
    y = df_clean["dropout"].values.astype(np.int64)

    cat_idxs = [X.columns.get_loc(c) for c in CAT_COLS]
    cat_dims  = [int(X[c].nunique()) + 1 for c in CAT_COLS]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X.values, y, X.index,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params(PARAMS)

        model = TabNetClassifier(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=4,
            n_d=PARAMS["n_d"],
            n_a=PARAMS["n_a"],
            n_steps=PARAMS["n_steps"],
            gamma=PARAMS["gamma"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": PARAMS["lr"]},
            mask_type="sparsemax",
            verbose=10,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=["auc"],
            max_epochs=PARAMS["max_epochs"],
            patience=PARAMS["patience"],
            batch_size=PARAMS["batch_size"],
            virtual_batch_size=PARAMS["virtual_batch_size"],
            weights=1,
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "recall":   recall_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred),
            "auc":      roc_auc_score(y_test, y_prob),
            "accuracy": accuracy_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        os.makedirs("outputs", exist_ok=True)
        model.save_model(MODEL_PATH)
        mlflow.log_artifact(f"{MODEL_PATH}.zip")

        run_id = run.info.run_id

    print(classification_report(y_test, y_pred, target_names=["유지", "이탈"]))
    print(f"run_id : {run_id}")

    predictions_df = pd.DataFrame({
        "id_student": df.loc[idx_test, "id_student"].values,
        "model_name": "tabnet",
        "predicted":  y_pred,
        "probability": y_prob,
        "run_id":     run_id,
    })
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM predictions WHERE model_name = 'tabnet'"))
        conn.commit()
    predictions_df.to_sql("predictions", engine, if_exists="append", index=False)
    print(f"predictions 저장 완료: {len(predictions_df)}건")

    return run_id


# ================================
# 추론 (Streamlit용)
# ================================
def load_model() -> TabNetClassifier:
    model = TabNetClassifier()
    model.load_model(f"{MODEL_PATH}.zip")
    return model


def predict(model: TabNetClassifier, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    반환: (predicted, probability)
    - predicted  : 0(유지) / 1(이탈) 배열
    - probability: 이탈 확률 배열
    """
    predicted = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    return predicted, probability


# ================================
# 직접 실행 시 학습
# ================================
if __name__ == "__main__":
    from sqlalchemy import create_engine

    engine = create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    train(engine)
