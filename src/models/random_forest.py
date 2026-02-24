"""
Random Forest 이탈 예측 모델
- train(engine) : 학습 + MLflow 로깅 + predictions 저장
- load_model()  : 저장된 모델 로드 (Streamlit용)
- predict()     : 추론 (Streamlit용)
"""
import os
import numpy as np
import joblib
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import randint

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

import mlflow
import mlflow.sklearn
from sqlalchemy import text

load_dotenv()

# ================================
# 상수
# ================================
EXPERIMENT   = 'random-forest-prediction-v2'
MODEL_NAME   = 'random_forest'
MODEL_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'random_forest_model.pkl')
RANDOM_STATE = 42
TEST_SIZE    = 0.2

IMD_ORDER = [
    '0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
    '50-60%', '60-70%', '70-80%', '80-90%', '90-100%',
]
AGE_ORDER = ['0-35', '35-55', '55<=']
EDU_ORDER = [
    'No Formal Quals',
    'Lower Than A Level',
    'A Level or Equivalent',
    'HE Qualification',
    'Post Graduate Qualification',
]

PARAM_DIST = {
    'n_estimators':      randint(100, 301),
    'max_depth':         [5, 8, 10, 12, 15],
    'min_samples_split': randint(2, 21),
    'min_samples_leaf':  randint(1, 11),
    'bootstrap':         [True, False],
    'class_weight':      [None, 'balanced'],
}


# ================================
# 전처리
# ================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=['id_student'], errors='ignore')

    for col, mp in [('gender', {'M': 1, 'F': 0}), ('disability', {'Y': 1, 'N': 0})]:
        if col in df.columns:
            df[col] = df[col].map(mp)

    enc = OrdinalEncoder(
        categories=[IMD_ORDER, AGE_ORDER, EDU_ORDER],
        handle_unknown='use_encoded_value',
        unknown_value=-1,
    )
    df[['imd_band', 'age_band', 'highest_education']] = enc.fit_transform(
        df[['imd_band', 'age_band', 'highest_education']]
    )

    for col in ['region', 'code_module', 'code_presentation']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


# ================================
# 학습
# ================================
def train(engine):
    df = pd.read_sql('SELECT * FROM students', engine)
    print(f'students : {df.shape}')

    id_arr = df['id_student'].values

    df_encoded = preprocess(df)
    X = df_encoded.drop(columns=['dropout'])
    y = df_encoded['dropout']

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index,
        test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    print(f'Train: {len(X_train):,}  |  Test: {len(X_test):,}')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=PARAM_DIST,
        n_iter=10,
        scoring='f1',
        cv=cv,
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name='random-forest') as run:
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred  = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            'f1':        f1_score(y_test, y_pred),
            'roc_auc':   roc_auc_score(y_test, y_proba),
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall':    recall_score(y_test, y_pred),
        }
        print('\n'.join(f'{k}: {v:.4f}' for k, v in metrics.items()))

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, 'model')

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        run_id = run.info.run_id

    print(f'MLflow run_id: {run_id}')

    # predictions 테이블 저장
    predictions_df = pd.DataFrame({
        'id_student': id_arr[idx_test],
        'model_name': MODEL_NAME,
        'predicted':  y_pred,
        'probability': y_proba,
        'run_id':     run_id,
    })
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM predictions WHERE model_name = :name"), {"name": MODEL_NAME})
        conn.commit()
    predictions_df.to_sql('predictions', engine, if_exists='append', index=False)
    print(f'predictions 저장 완료: {len(predictions_df)}건')

    return run_id


# ================================
# 추론 (Streamlit용)
# ================================
def load_model() -> RandomForestClassifier:
    return joblib.load(MODEL_PATH)


def predict(model: RandomForestClassifier, X: np.ndarray) -> tuple:
    """
    반환: (predicted, probability)
    - predicted  : 0(유지) / 1(이탈) 배열
    - probability: 이탈 확률 배열
    """
    predicted   = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    return predicted, probability


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
