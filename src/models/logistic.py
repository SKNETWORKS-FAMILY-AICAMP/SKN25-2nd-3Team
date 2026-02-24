"""
Logistic Regression 이탈 예측 모델
- train(engine) : 학습 + MLflow 로깅 + predictions 저장
"""
import os
import warnings
warnings.filterwarnings('ignore')

import joblib
import tempfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)
import mlflow
import mlflow.sklearn
from sqlalchemy import text

load_dotenv()

# ================================
# 상수
# ================================
EXPERIMENT   = 'student_dropout_logistic_v2'
MODEL_NAME   = 'logistic_dropout'
MODEL_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'logistic_model.pkl')
RANDOM_STATE = 42
TEST_SIZE    = 0.2

ORDINAL_FEATURES = ['imd_band', 'age_band', 'highest_education']
NOMINAL_FEATURES = ['region', 'code_module', 'code_presentation']
BINARY_FEATURES  = ['gender', 'disability']

ORDINAL_CATEGORIES = [
    ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
     '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
    ['0-35', '35-55', '55<='],
    ['No Formal Quals', 'Lower Than A Level', 'A Level or Equivalent',
     'HE Qualification', 'Post Graduate Qualification'],
]

LR_PARAMS = {
    'solver':       'liblinear',
    'max_iter':     1000,
    'class_weight': 'balanced',
    'C':            0.5,
}


# ================================
# 전처리
# ================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['gender']     = df['gender'].map({'M': 1, 'F': 0})
    df['disability'] = df['disability'].map({'Y': 1, 'N': 0})
    df['dropout']    = df['dropout'].astype(int)
    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    used = set(ORDINAL_FEATURES + NOMINAL_FEATURES + BINARY_FEATURES)
    numeric_features = [c for c in X.columns if c not in used]

    ordinal_features = [c for c in ORDINAL_FEATURES if c in X.columns]
    nominal_features = [c for c in NOMINAL_FEATURES if c in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoder(
                categories=ORDINAL_CATEGORIES,
                handle_unknown='use_encoded_value',
                unknown_value=-1,
            ), ordinal_features),
            ('nom', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=True,
            ), nominal_features),
            ('num', StandardScaler(with_mean=False), numeric_features),
        ],
        remainder='drop',
        sparse_threshold=1.0,
    )

    return Pipeline([
        ('preprocess', preprocessor),
        ('model', LogisticRegression(**LR_PARAMS)),
    ])


# ================================
# 학습
# ================================
def train(engine):
    df = pd.read_sql('SELECT * FROM students', engine)
    print(f'students : {df.shape}')

    id_arr = df['id_student'].values

    df = preprocess(df)
    X = df.drop(columns=['id_student', 'dropout'])
    y = df['dropout']

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index,
        test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    print(f'Train: {len(X_train):,}  |  Test: {len(X_test):,}')

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob),
    }
    print('\n'.join(f'{k}: {v:.4f}' for k, v in metrics.items()))

    # Confusion Matrix 저장
    os.makedirs('outputs/logistic', exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=1, linecolor='white')
    ax.set_xticklabels(['Not Dropout', 'Dropout'])
    ax.set_yticklabels(['Not Dropout', 'Dropout'], rotation=0)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - Logistic Regression')
    fig.tight_layout()
    fig.savefig('outputs/logistic/confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # MLflow 로깅
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(EXPERIMENT)

    # 로컬 모델 저장 (Streamlit fallback 용도)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    with mlflow.start_run(run_name='logreg') as run:
        mlflow.log_params(LR_PARAMS)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact('outputs/logistic/confusion_matrix.png')

        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'logistic_model.pkl')
            joblib.dump(pipeline, model_path)
            mlflow.log_artifact(model_path, artifact_path='model')

        run_id = run.info.run_id

    print(f'MLflow run_id: {run_id}')

    # predictions 테이블 저장
    predictions_df = pd.DataFrame({
        'id_student': id_arr[idx_test],
        'model_name': MODEL_NAME,
        'predicted':  y_pred,
        'probability': y_prob,
        'run_id':     run_id,
    })
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM predictions WHERE model_name = :name"), {"name": MODEL_NAME})
        conn.commit()
    predictions_df.to_sql('predictions', engine, if_exists='append', index=False)
    print(f'predictions 저장 완료: {len(predictions_df)}건')

    return run_id


# ================================
# 직접 실행 시 학습
# ================================
if __name__ == '__main__':
    from sqlalchemy import create_engine

    engine = create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    train(engine)
