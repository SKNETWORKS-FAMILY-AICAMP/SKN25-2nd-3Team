"""
predictions 테이블에 섞여 저장된 사용자 예측 데이터를
live_predictions 테이블로 이관하는 스크립트.

실행:
  python scripts/migrate_live_predictions.py
"""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()


def get_engine():
    return create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


LIVE_MODEL_NAMES = [
    "Random Forest",
    "Logistic Regression",
    "XGBoost",
    "CatBoost",
]

MODEL_PARAMS = {
    "m1": LIVE_MODEL_NAMES[0],
    "m2": LIVE_MODEL_NAMES[1],
    "m3": LIVE_MODEL_NAMES[2],
    "m4": LIVE_MODEL_NAMES[3],
}


CREATE_LIVE_TABLE_SQL = """
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


COUNT_CANDIDATES_SQL = """
SELECT COUNT(*)
FROM predictions
WHERE model_name IN (:m1, :m2, :m3, :m4)
"""


INSERT_SQL = """
INSERT INTO live_predictions (id_student, model_name, predicted, probability, run_id)
SELECT p.id_student, p.model_name, p.predicted, p.probability, p.run_id
FROM predictions p
WHERE p.model_name IN (:m1, :m2, :m3, :m4)
"""


DELETE_SQL = """
DELETE FROM predictions
WHERE model_name IN (:m1, :m2, :m3, :m4)
"""


def main():
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text(CREATE_LIVE_TABLE_SQL))

        candidates = conn.execute(
            text(COUNT_CANDIDATES_SQL),
            MODEL_PARAMS,
        ).scalar_one()

        if candidates == 0:
            print("이관 대상이 없습니다. (predictions 내 사용자 저장 데이터 0건)")
            return

        inserted = conn.execute(
            text(INSERT_SQL),
            MODEL_PARAMS,
        ).rowcount

        deleted = conn.execute(
            text(DELETE_SQL),
            MODEL_PARAMS,
        ).rowcount

    print(f"이관 완료: 후보 {candidates}건 / live_predictions 추가 {inserted}건 / predictions 삭제 {deleted}건")


if __name__ == "__main__":
    main()
