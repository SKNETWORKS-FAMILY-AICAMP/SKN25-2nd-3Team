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
DB_PORT = os.getenv("DB_PORT", "3307")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ================================
# 테이블 생성
# ================================
DROP_STUDENT_VLE = "DROP TABLE IF EXISTS studentVle;"
DROP_STUDENTS    = "DROP TABLE IF EXISTS students;"
DROP_PREDICTIONS = "DROP TABLE IF EXISTS predictions;"
DROP_CLUSTERS    = "DROP TABLE IF EXISTS clusters;"
DROP_VLE         = "DROP TABLE IF EXISTS vle;"

CREATE_STUDENTS = """
CREATE TABLE students (
    id_student                 INT,
    code_module                VARCHAR(10),
    code_presentation          VARCHAR(20),
    gender                     VARCHAR(10),
    region                     VARCHAR(100),
    highest_education          VARCHAR(100),
    imd_band                   VARCHAR(20),
    age_band                   VARCHAR(20),
    num_of_prev_attempts       INT,
    studied_credits            INT,
    disability                 VARCHAR(5),
    total_clicks               FLOAT,
    active_days                FLOAT,
    unique_resources           FLOAT,
    num_forum                  FLOAT,
    num_quiz                   FLOAT,
    avg_score                  FLOAT,
    num_assess_attempted       FLOAT,
    total_weight               FLOAT,
    module_presentation_length INT,
    dropout                    TINYINT
);
"""

CREATE_PREDICTIONS = """
CREATE TABLE predictions (
    id_student  INT,
    model_name  VARCHAR(50),
    predicted   TINYINT,
    probability FLOAT,
    run_id      VARCHAR(100)
);
"""

CREATE_CLUSTERS = """
CREATE TABLE clusters (
    id_student   INT,
    cluster_id   INT,
    dropout_rate FLOAT
);
"""

CREATE_VLE = """
CREATE TABLE vle (
    id_site           INT,
    code_module       VARCHAR(10),
    code_presentation VARCHAR(20),
    activity_type     VARCHAR(45),
    week_from         INT,
    week_to           INT
);
"""

CREATE_STUDENT_VLE = """
CREATE TABLE studentVle (
    code_module       VARCHAR(10),
    code_presentation VARCHAR(20),
    id_student        INT,
    id_site           INT,
    date              INT,
    sum_click         INT
);
"""

with engine.connect() as conn:
    conn.execute(text(DROP_STUDENT_VLE))
    conn.execute(text(DROP_STUDENTS))
    conn.execute(text(DROP_PREDICTIONS))
    conn.execute(text(DROP_CLUSTERS))
    conn.execute(text(DROP_VLE))
    conn.execute(text(CREATE_STUDENTS))
    conn.execute(text(CREATE_PREDICTIONS))
    conn.execute(text(CREATE_CLUSTERS))
    conn.execute(text(CREATE_VLE))
    conn.execute(text(CREATE_STUDENT_VLE))
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
    "id_student", "code_module", "code_presentation",
    "gender", "region", "highest_education", "imd_band",
    "age_band", "num_of_prev_attempts", "studied_credits", "disability",
    "total_clicks", "active_days", "unique_resources", "num_forum",
    "num_quiz", "avg_score", "num_assess_attempted", "total_weight",
    "module_presentation_length", "dropout"
]
df_students = df[students_cols].copy()

# 데이터 품질 수정
df_students["imd_band"] = df_students["imd_band"].replace("10-20", "10-20%")  # % 누락
df_students["imd_band"] = df_students["imd_band"].replace("0", None)           # 결측값
df_students["highest_education"] = df_students["highest_education"].replace(
    "No Formal quals", "No Formal Quals"
)

df_students.to_sql("students", engine, if_exists="append", index=False)
print(f"students 적재 완료: {len(df_students)}행")

# vle
df_vle = pd.read_csv("data/vle.csv")
df_vle.to_sql("vle", engine, if_exists="append", index=False)
print(f"vle 적재 완료: {len(df_vle)}행")

# studentVle (청크 처리)
CHUNK = 50_000
total = 0
for chunk in pd.read_csv("data/studentVle.csv", chunksize=CHUNK):
    chunk.to_sql("studentVle", engine, if_exists="append", index=False)
    total += len(chunk)
    print(f"  studentVle: {total:,}행 적재 중...", end="\r")
print(f"\nstudentVle 적재 완료: {total:,}행")

# 확인
with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM students")).scalar()
    dropout_count = conn.execute(text("SELECT SUM(dropout) FROM students")).scalar()
    print(f"students: {count}행 (이탈: {int(dropout_count)}명)")
    print(f"vle: {conn.execute(text('SELECT COUNT(*) FROM vle')).scalar()}행")
    print(f"studentVle: {conn.execute(text('SELECT COUNT(*) FROM studentVle')).scalar()}행")
