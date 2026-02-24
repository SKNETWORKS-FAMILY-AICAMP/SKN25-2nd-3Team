import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


@st.cache_resource
def get_engine():
    return create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


@st.cache_data(ttl=60)
def load_predictions():
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT p.id_student, p.model_name, p.predicted, p.probability, p.run_id,
               s.gender, s.age_band, s.region, s.highest_education,
               s.imd_band, s.studied_credits, s.dropout
        FROM predictions p
        LEFT JOIN (
            SELECT id_student,
                   ANY_VALUE(gender)            AS gender,
                   ANY_VALUE(age_band)          AS age_band,
                   ANY_VALUE(region)            AS region,
                   ANY_VALUE(highest_education) AS highest_education,
                   ANY_VALUE(imd_band)          AS imd_band,
                   ANY_VALUE(studied_credits)   AS studied_credits,
                   MAX(dropout)                 AS dropout
            FROM students
            GROUP BY id_student
        ) s ON p.id_student = s.id_student
        """,
        engine,
    )


@st.cache_data(ttl=60)
def load_clusters():
    engine = get_engine()
    return pd.read_sql(
        """
        SELECT
            c.id_student,
            c.cluster_id,
            s.avg_score,
            s.active_days,
            s.total_clicks,
            s.dropout
        FROM clusters c
        JOIN (
            SELECT id_student,
                   ANY_VALUE(avg_score)    AS avg_score,
                   ANY_VALUE(active_days)  AS active_days,
                   ANY_VALUE(total_clicks) AS total_clicks,
                   MAX(dropout)            AS dropout
            FROM students
            GROUP BY id_student
        ) s ON c.id_student = s.id_student
        """,
        engine,
    )
