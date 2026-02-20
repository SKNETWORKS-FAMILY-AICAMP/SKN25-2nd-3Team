"""
MySQL 스키마 생성 + 데이터 적재 스크립트
실행: python scripts/init_db.py
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ================================
# 테이블 생성
# ================================
CREATE_STUDENTS = """
CREATE TABLE IF NOT EXISTS students (
    id_student          INT,
    gender              VARCHAR(10),
    region              VARCHAR(100),
    highest_education   VARCHAR(100),
    imd_band            VARCHAR(20),
    age_band            VARCHAR(20),
    num_of_prev_attempts INT,
    studied_credits     INT,
    disability          VARCHAR(5),
    total_clicks        FLOAT,
    active_days         FLOAT,
    unique_resources    FLOAT,
    num_forum           FLOAT,
    num_quiz            FLOAT,
    avg_score           FLOAT,
    num_assess_attempted FLOAT,
    total_weight        FLOAT,
    module_presentation_length INT,
    dropout             TINYINT
);
"""

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id_student  INT,
    model_name  VARCHAR(50),
    predicted   TINYINT,
    probability FLOAT,
    run_id      VARCHAR(100)
);
"""

CREATE_CLUSTERS = """
CREATE TABLE IF NOT EXISTS clusters (
    id_student   INT,
    cluster_id   INT,
    dropout_rate FLOAT
);
"""

with engine.connect() as conn:
    conn.execute(text(CREATE_STUDENTS))
    conn.execute(text(CREATE_PREDICTIONS))
    conn.execute(text(CREATE_CLUSTERS))
    conn.commit()
    print("테이블 생성 완료")

# ================================
# 데이터 적재
# ================================
df = pd.read_csv("data/final_dataset.csv")

# final_result → dropout (Withdrawn=1, 나머지=0)
df["dropout"] = (df["final_result"] == "Withdrawn").astype(int)

# students 테이블 컬럼만 선택
students_cols = [
    "id_student", "gender", "region", "highest_education", "imd_band",
    "age_band", "num_of_prev_attempts", "studied_credits", "disability",
    "total_clicks", "active_days", "unique_resources", "num_forum",
    "num_quiz", "avg_score", "num_assess_attempted", "total_weight",
    "module_presentation_length", "dropout"
]
df_students = df[students_cols]

df_students.to_sql("students", engine, if_exists="replace", index=False)
print(f"데이터 적재 완료: {len(df_students)}행")

# 확인
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM students"))
    count = result.scalar()
    dropout_count = conn.execute(text("SELECT SUM(dropout) FROM students")).scalar()
    print(f"students 테이블: {count}행 (이탈: {int(dropout_count)}명)")
