"""
Student Dropout — XGBoost + SHAP + MLflow
Flow:
  1. students + studentVle 로드 (MySQL, 청크 방식)
  2. studentVle CUTOFF_DAY 기준 집계
  3. 전처리 & 합치기
  4. 전처리 데이터 MySQL 업로드
  5. XGBoost 학습 (RandomizedSearchCV 튜닝)
  6. MLflow 로깅 & 모델 등록
  7. predictions 테이블 DB 저장
  8. SHAP 시각화
"""

import warnings
warnings.filterwarnings('ignore')

import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 서버 환경 — 디스플레이 없이 저장
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from sqlalchemy import create_engine, Float, Integer, SmallInteger, String, text

# ── .env 로드 ─────────────────────────────────────────────────────────────────
load_dotenv('/home/ict/work/.env')

DB_URI       = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
MLFLOW_URI   = os.getenv('MLFLOW_TRACKING_URI')
EXPERIMENT   = 'xgboost_stu_vle'
MODEL_NAME   = 'xgboost_dropout'
OUTPUT_TABLE = 'preprocessed_students'
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CUTOFF_DAY   = 60
OUTPUT_DIR   = '/home/ict/work/outputs/xgboost'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('✅ Imports OK')
print(f'DB     : {os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}')
print(f'MLflow : {MLFLOW_URI}')
print(f'Cutoff : {CUTOFF_DAY}일')


# ── 1. 데이터 로드 ────────────────────────────────────────────────────────────
engine = create_engine(DB_URI)

# students (작은 테이블 — 바로 로드)
df_students = pd.read_sql('SELECT * FROM students', con=engine)
print(f'students   : {df_students.shape}')

# studentVle (1천만행 — 청크로 분할 로드)
chunks, offset, batch = [], 0, 100_000
while True:
    query = f'SELECT * FROM studentVle LIMIT {batch} OFFSET {offset}'
    with engine.connect() as conn:
        df_chunk = pd.read_sql(query, con=conn)
    if df_chunk.empty:
        break
    chunks.append(df_chunk)
    offset += batch
    print(f'  로드 중... {offset:,}행 ({min(offset, 10_655_280)/10_655_280*100:.1f}%)')

df_vle = pd.concat(chunks, ignore_index=True)
print(f'studentVle : {df_vle.shape}')


# ── 2. VLE 집계 (CUTOFF_DAY 기준, 누수 제거) ─────────────────────────────────
df_vle_cut = df_vle[df_vle['date'] <= CUTOFF_DAY].copy()

vle_features = df_vle_cut.groupby(
    ['id_student', 'code_module', 'code_presentation']
).agg(
    vle_total_clicks       = ('sum_click', 'sum'),
    vle_active_days        = ('date',      'nunique'),
    vle_unique_resources   = ('id_site',   'nunique'),
    vle_avg_clicks_per_day = ('sum_click', 'mean'),
    vle_max_clicks         = ('sum_click', 'max'),
    vle_std_clicks         = ('sum_click', 'std'),
).reset_index()
vle_features['vle_std_clicks'] = vle_features['vle_std_clicks'].fillna(0)
print(f'VLE features shape: {vle_features.shape}')

# 누수 확인
tmp = df_students[['id_student','code_module','code_presentation','dropout']].merge(
    vle_features, on=['id_student','code_module','code_presentation'], how='left'
)
vle_cols = ['vle_total_clicks','vle_active_days','vle_unique_resources']
tmp[vle_cols] = tmp[vle_cols].fillna(0)
check = tmp.groupby('dropout')[vle_cols].mean().round(2)
ratio = check.loc[0,'vle_total_clicks'] / max(check.loc[1,'vle_total_clicks'], 1)
print(f'\n=== {CUTOFF_DAY}일 기준 dropout별 평균 ===\n{check}')
print(f'vle_total_clicks 비율(Stay/Dropout): {ratio:.1f}x  ※ 5x 이상이면 누수 의심')


# ── 3. Feature 합치기 ─────────────────────────────────────────────────────────
LEAKAGE_COLS = [
    'total_clicks', 'active_days', 'unique_resources',
    'num_forum', 'num_quiz', 'avg_score', 'num_assess_attempted', 'total_weight'
]
df_base   = df_students.drop(columns=LEAKAGE_COLS, errors='ignore')
df_merged = df_base.merge(
    vle_features, on=['id_student','code_module','code_presentation'], how='left'
)
all_vle_cols = [
    'vle_total_clicks','vle_active_days','vle_unique_resources',
    'vle_avg_clicks_per_day','vle_max_clicks','vle_std_clicks'
]
for col in all_vle_cols:
    df_merged[col] = df_merged[col].fillna(0)
print(f'최종 shape: {df_merged.shape}')


# ── 4. 전처리 ─────────────────────────────────────────────────────────────────
TARGET    = 'dropout'
DROP_COLS = ['id_student']

BINARY_MAP = {
    'gender':     {'M': 1, 'F': 0},
    'disability': {'Y': 1, 'N': 0}
}
ORDINAL_COLS = {
    'imd_band': [
        '0-10%','10-20%','20-30%','30-40%','40-50%',
        '50-60%','60-70%','70-80%','80-90%','90-100%'
    ],
    'age_band': ['0-35', '35-55', '55<='],
    'highest_education': [
        'No Formal Quals', 'Lower Than A Level',
        'A Level or Equivalent', 'HE Qualification', 'Post Graduate Qualification'
    ]
}
LABEL_COLS = ['region', 'code_module', 'code_presentation']


def preprocess(df: pd.DataFrame):
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
                unknown_value=-1
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


X, y, encoders = preprocess(df_merged)


# ── 5. 전처리 데이터 MySQL 업로드 ─────────────────────────────────────────────
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
    'vle_avg_clicks_per_day':     Float(),
    'vle_max_clicks':             Float(),
    'vle_std_clicks':             Float(),
    'dropout':                    SmallInteger(),
}
df_out.to_sql(
    name=OUTPUT_TABLE, con=engine, if_exists='replace',
    index=False, chunksize=5000, dtype=dtype_map,
)
print(f"✅ '{OUTPUT_TABLE}' 업로드 완료 — {len(df_out):,} rows")


# ── 6. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f'Train: {len(X_train):,}  |  Test: {len(X_test):,}')


# ── 7. XGBoost 튜닝 (RandomizedSearchCV) ─────────────────────────────────────
param_dist = {
    'n_estimators':     [500, 700, 1000],
    'max_depth':        [4, 5, 6],
    'learning_rate':    [0.01, 0.05],
    'subsample':        [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'min_child_weight': [1, 3, 5],
    'gamma':            [0, 0.1, 0.3],
}
search = RandomizedSearchCV(
    XGBClassifier(
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        eval_metric='auc',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1',
    cv=3,
    random_state=RANDOM_STATE,
    verbose=1,
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f'Best params: {search.best_params_}')

y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
auc    = roc_auc_score(y_test, y_prob)
f1     = f1_score(y_test, y_pred)
print(f'\nAUC : {auc:.4f}')
print(f'F1  : {f1:.4f}')
print(classification_report(y_test, y_pred, target_names=['Stay','Dropout']))


# ── 8. MLflow 로깅 & 모델 등록 ───────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name='xgboost') as run:
    RUN_ID = run.info.run_id
    print(f'MLflow run_id: {RUN_ID}')

    mlflow.log_params({**search.best_params_, 'cutoff_day': CUTOFF_DAY})
    mlflow.log_metrics({'auc': auc, 'f1': f1})

    # Confusion Matrix 저장
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=['Stay', 'Dropout']
    ).plot(ax=ax)
    ax.set_title(f'Confusion Matrix - tuned (cutoff={CUTOFF_DAY}일)')
    fig.savefig(f'{OUTPUT_DIR}/confusion_matrix_tuned.png', bbox_inches='tight')
    mlflow.log_artifact(f'{OUTPUT_DIR}/confusion_matrix_tuned.png')
    plt.close()

    # 모델 저장
    signature = infer_signature(X_train, best_model.predict(X_train))
    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow.xgboost.save_model(
            xgb_model=best_model, path=tmp_dir,
            signature=signature, input_example=X_train.iloc[:5],
        )
        mlflow.log_artifacts(tmp_dir, artifact_path='model')

# 모델 등록
client  = MlflowClient()
run_uri = f'runs:/{RUN_ID}/model'
try:
    client.create_registered_model(MODEL_NAME)
except Exception:
    pass
client.create_model_version(name=MODEL_NAME, source=run_uri, run_id=RUN_ID)
print(f"✅ Model registered as '{MODEL_NAME}'")


# ── 9. predictions 테이블 DB 저장 ────────────────────────────────────────────
# id_student는 df_merged 기준 인덱스와 맞춰서 복원
id_students_all = df_merged['id_student'].values
_, X_test_idx, _, _ = train_test_split(
    np.arange(len(X)), y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
df_pred = pd.DataFrame({
    'id_student': id_students_all[X_test_idx],
    'model_name': MODEL_NAME,
    'predicted':  y_pred,
    'probability': y_prob.round(4),
    'run_id':     RUN_ID,
})
df_pred.to_sql(
    name='predictions', con=engine, if_exists='replace',
    index=False, chunksize=5000,
    dtype={
        'id_student':  Integer(),
        'model_name':  String(50),
        'predicted':   SmallInteger(),
        'probability': Float(),
        'run_id':      String(50),
    }
)
print(f"✅ 'predictions' 업로드 완료 — {len(df_pred):,} rows")


# ── 10. SHAP 시각화 & MLflow 저장 ────────────────────────────────────────────
explainer = shap.TreeExplainer(best_model)
shap_vals = explainer(X_test)
print('SHAP values computed ✅')

# Beeswarm
shap.plots.beeswarm(shap_vals, max_display=20, show=False)
plt.title(f'SHAP Beeswarm (cutoff={CUTOFF_DAY}일)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_beeswarm.png', bbox_inches='tight')
plt.close()

# Bar
shap.plots.bar(shap_vals, max_display=20, show=False)
plt.title(f'SHAP Bar (cutoff={CUTOFF_DAY}일)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_bar.png', bbox_inches='tight')
plt.close()

# MLflow에 SHAP 아티팩트 저장
with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_artifact(f'{OUTPUT_DIR}/shap_beeswarm.png')
    mlflow.log_artifact(f'{OUTPUT_DIR}/shap_bar.png')
print('✅ SHAP artifacts logged to MLflow')
print('\n🎉 전체 파이프라인 완료!')