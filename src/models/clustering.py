"""
KMeans 군집 분석 모델
- train(engine) : 군집화 + MLflow 로깅 + clusters/kmeans_metrics 저장
"""
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import mlflow

load_dotenv()

# ================================
# 상수
# ================================
K        = 3
FEATURES = ['avg_score', 'active_days', 'total_clicks']
EXPERIMENT = 'student-clustering-analysis'
RUN_NAME   = 'kmeans'


# ================================
# 학습
# ================================
def train(engine):
    df = pd.read_sql(
        "SELECT id_student, avg_score, active_days, total_clicks, dropout FROM students",
        engine,
    )
    print(f'students : {df.shape}')

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[FEATURES])

    model = KMeans(n_clusters=K, random_state=42)
    df['cluster_id'] = model.fit_predict(df_scaled)

    cluster_dropout_map = df.groupby('cluster_id')['dropout'].mean().to_dict()
    df['dropout_rate'] = df['cluster_id'].map(cluster_dropout_map)

    # MLflow 로깅
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_param('k', K)
        mlflow.log_metric('inertia', model.inertia_)
        run_id = run.info.run_id

    print(f'MLflow run_id: {run_id}')

    # clusters 테이블 저장 (init_db 스키마: id_student, cluster_id, dropout_rate)
    df[['id_student', 'cluster_id', 'dropout_rate']].to_sql(
        'clusters', engine, if_exists='replace', index=False,
    )
    print(f'clusters 저장 완료: {len(df)}건')

    # kmeans_metrics 테이블 저장 (Elbow 차트용)
    elbow_data = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(df_scaled)
        elbow_data.append({'k': k, 'inertia': km.inertia_})

    pd.DataFrame(elbow_data).to_sql(
        'kmeans_metrics', engine, if_exists='replace', index=False,
    )
    print('kmeans_metrics 저장 완료')

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
