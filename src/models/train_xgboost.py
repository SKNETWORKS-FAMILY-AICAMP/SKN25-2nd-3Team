"""
XGBoost 이탈 예측 모델
Student Dropout — XGBoost + SHAP + MLflow

Flow:
1. students + studentVle 로드 (MySQL)
2. studentVle CUTOFF_DAY 기준 집계 → 누수 없는 조기 개입 feature
3. 전처리 & 합치기
4. 전처리 데이터 MySQL 업로드
5. XGBoost 학습 + MLflow 로깅
6. SHAP 시각화
"""
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
from sqlalchemy import create_engine, Float, Integer, SmallInteger

load_dotenv()

# ── 설정 ────────────────────────────────────────────────
DB_URI = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
MLFLOW_URI   = os.getenv('MLFLOW_TRACKING_URI')
EXPERIMENT   = 'student_dropout_vle'
MODEL_NAME   = 'xgboost_dropout'
OUTPUT_TABLE = 'preprocessed_students'
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CUTOFF_DAY   = 30   # 30 or 60 — 몇 일치 VLE 데이터를 쓸지

TARGET    = 'dropout'
DROP_COLS = ['id_student']

BINARY_MAP = {
    'gender':     {'M': 1, 'F': 0},
    'disability': {'Y': 1, 'N': 0},
}
ORDINAL_COLS = {
    'imd_band': [
        '0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
        '50-60%', '60-70%', '70-80%', '80-90%', '90-100%',
    ],
    'age_band': ['0-35', '35-55', '55<='],
    'highest_education': [
        'No Formal Quals',
        'Lower Than A Level',
        'A Level or Equivalent',
        'HE Qualification',
        'Post Graduate Qualification',
    ],
}
LABEL_COLS = ['region', 'code_module', 'code_presentation']
LEAKAGE_COLS = [
    'total_clicks', 'active_days', 'unique_resources',
    'num_forum', 'num_quiz',
    'avg_score', 'num_assess_attempted', 'total_weight',
]

XGB_PARAMS = {
    'n_estimators':     500,
    'max_depth':        6,
    'learning_rate':    0.05,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'eval_metric':      'auc',
    'random_state':     RANDOM_STATE,
    'n_jobs':           -1,
}
# ────────────────────────────────────────────────────────


def load_data(engine):
    df_students = pd.read_sql('SELECT * FROM students', con=engine)
    df_vle      = pd.read_sql('SELECT * FROM studentVle', con=engine)
    print(f'students   : {df_students.shape}')
    print(f'studentVle : {df_vle.shape}')
    return df_students, df_vle


def build_vle_features(df_vle):
    df_vle_cut = df_vle[df_vle['date'] <= CUTOFF_DAY].copy()
    vle_features = df_vle_cut.groupby(
        ['id_student', 'code_module', 'code_presentation']
    ).agg(
        vle_total_clicks     = ('sum_click', 'sum'),
        vle_active_days      = ('date', 'nunique'),
        vle_unique_resources = ('id_site', 'nunique'),
    ).reset_index()
    print(f'VLE features shape: {vle_features.shape}')
    return vle_features


def merge_features(df_students, vle_features):
    df_base = df_students.drop(columns=LEAKAGE_COLS, errors='ignore')
    df_merged = df_base.merge(
        vle_features,
        on=['id_student', 'code_module', 'code_presentation'],
        how='left',
    )
    for col in ['vle_total_clicks', 'vle_active_days', 'vle_unique_resources']:
        df_merged[col] = df_merged[col].fillna(0)
    print(f'최종 shape: {df_merged.shape}')
    return df_merged


def preprocess(df):
    df = df.copy()
    y  = df[TARGET].astype(int)
    df = df.drop(columns=DROP_COLS + [TARGET], errors='ignore')

    encoders = {}

    for col, mapping in BINARY_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    for col, order in ORDINAL_COLS.items():
        if col in df.columns:
            enc = OrdinalEncoder(
                categories=[order],
                handle_unknown='use_encoded_value',
                unknown_value=-1,
            )
            df[col] = enc.fit_transform(df[[col]])
            encoders[f'ord_{col}'] = enc

    for col in LABEL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[f'le_{col}'] = le

    print(f'X shape: {df.shape}  |  dropout rate: {y.mean():.3f}')
    return df, y, encoders


def upload_preprocessed(X, y, engine):
    df_out = X.copy()
    df_out[TARGET] = y.values
    dtype_map = {
        'num_of_prev_attempts':       Integer(),
        'studied_credits':            Integer(),
        'module_presentation_length': Integer(),
        'gender':                     SmallInteger(),
        'disability':                 SmallInteger(),
        'imd_band':                   SmallInteger(),
        'age_band':                   SmallInteger(),
        'highest_education':          SmallInteger(),
        'region':                     SmallInteger(),
        'code_module':                SmallInteger(),
        'code_presentation':          SmallInteger(),
        'vle_total_clicks':           Float(),
        'vle_active_days':            Float(),
        'vle_unique_resources':       Float(),
        'dropout':                    SmallInteger(),
    }
    df_out.to_sql(
        name      = OUTPUT_TABLE,
        con       = engine,
        if_exists = 'replace',
        index     = False,
        chunksize = 5000,
        dtype     = dtype_map,
    )
    print(f"'{OUTPUT_TABLE}' 업로드 완료 — {len(df_out):,} rows")


def train():
    engine = create_engine(DB_URI)

    df_students, df_vle = load_data(engine)
    vle_features = build_vle_features(df_vle)
    df_merged = merge_features(df_students, vle_features)

    X, y, _ = preprocess(df_merged)
    upload_preprocessed(X, y, engine)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    print(f'Train: {len(X_train):,}  |  Test: {len(X_test):,}')

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    params = {**XGB_PARAMS, 'scale_pos_weight': (y == 0).sum() / (y == 1).sum()}

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f'MLflow run_id: {run_id}')

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_prob)
        f1  = f1_score(y_test, y_pred)

        print(f'\nAUC : {auc:.4f}')
        print(f'F1  : {f1:.4f}')
        print(classification_report(y_test, y_pred, target_names=['Stay', 'Dropout']))

        mlflow.log_params({**params, 'cutoff_day': CUTOFF_DAY})
        mlflow.log_metrics({'auc': auc, 'f1': f1})

        # Confusion Matrix
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred),
            display_labels=['Stay', 'Dropout'],
        ).plot(ax=ax)
        ax.set_title(f'Confusion Matrix (cutoff={CUTOFF_DAY}일)')
        os.makedirs('outputs/xgboost', exist_ok=True)
        fig.savefig('outputs/xgboost/confusion_matrix.png', bbox_inches='tight')
        mlflow.log_artifact('outputs/xgboost/confusion_matrix.png')
        plt.close()

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(
            xgb_model             = model,
            artifact_path         = 'model',
            signature             = signature,
            registered_model_name = MODEL_NAME,
            input_example         = X_train.iloc[:5],
        )
        print(f"Model registered as '{MODEL_NAME}'")

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(X_test)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(shap_vals, max_display=20, show=False)
        fig.savefig('outputs/xgboost/shap_summary.png', bbox_inches='tight')
        mlflow.log_artifact('outputs/xgboost/shap_summary.png')
        plt.close()
        print('SHAP artifact logged to MLflow')

    return model


if __name__ == '__main__':
    train()
