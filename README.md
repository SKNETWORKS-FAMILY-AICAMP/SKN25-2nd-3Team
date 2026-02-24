# OULAD Dropout Analytics

OULAD(Open University Learning Analytics Dataset) 기반 학습 이탈 예측 및 군집 분석 대시보드 프로젝트입니다.  
MySQL + MLflow + Streamlit + Docker Compose 구조로 운영됩니다.

---

## What This Repo Delivers

- 이탈 예측 실시간 추론 (Logistic Regression, Random Forest, XGBoost)
- 예측 결과 분석 대시보드 (모델별 성능/분포/그룹 비교)
- 학습자 군집 분석 대시보드 (KMeans, 군집별 프로파일 해석)
- MLflow 기반 실험 추적 + MySQL 예측 저장

---

## ERD

아래 이미지는 현재 코드 기준 최종 DB 구조입니다.

<img width="933" height="1098" alt="Untitled" src="https://github.com/user-attachments/assets/e4c6e689-e185-4d09-852d-f73ed562c5b9" />

---

## Architecture

```text
Docker Compose
├─ MySQL        :3307
├─ MLflow       :5001
├─ Jupyter      :9000
└─ Streamlit    :8501
```

Streamlit app:
- `src/app.py`
- `src/pages/01_dropout_prediction.py`
- `src/pages/02_clustering_analysis.py`

---

## Repository Structure

```text
src/
├── app.py
├── common_data.py
├── models/
│   ├── clustering.py
│   ├── logistic.py
│   ├── random_forest.py
│   ├── tabnet.py
│   └── xgboost.py
└── pages/
    ├── 01_dropout_prediction.py
    └── 02_clustering_analysis.py

scripts/
└── init_db.py
```

---

## Quick Start

### 1) Environment

`.env`에 DB/MLflow 접속 정보를 설정합니다.

필수 변수 예시:
- `DB_USER`
- `DB_PASSWORD`
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `MLFLOW_TRACKING_URI`

### 2) Run services

```bash
docker compose up -d
```

### 3) Initialize DB (최초 1회)

```bash
python scripts/init_db.py
```

### 4) Access

- Jupyter: `http://localhost:9000`
- Streamlit: `http://localhost:8501`

---

## PyTorch (TabNet) Platform Guide

TabNet 학습을 로컬에서 실행할 때는 OS/가속기 환경에 맞춰 PyTorch를 설치하세요.

- Windows + NVIDIA (CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pytorch-tabnet
```

- Apple Silicon (M1/M2/M3):
```bash
pip install torch torchvision torchaudio
pip install pytorch-tabnet
```

참고:
- Docker 컨테이너 기준 기본 의존성은 `docker_env/requirements.txt`에서 관리됩니다.

---

## Model Training & Logging

각 모델 스크립트 실행 시 학습 + MLflow 로깅 + `predictions` 테이블 적재를 수행합니다.

```bash
python src/models/logistic.py
python src/models/random_forest.py
python src/models/xgboost.py
python src/models/tabnet.py
python src/models/clustering.py
```

---

## Streamlit Pages

### 1) 이탈 예측

- 실시간 추론: Logistic Regression / Random Forest / XGBoost
- 입력 모드: 간소화 입력 / 전체 입력
- 저장 옵션: `predictions` 테이블에 실시간 예측 저장 가능

### 2) 군집 분석

- 군집 KPI, 분포 시각화, 군집 프로파일 비교
- 군집별 해석 카드(참여도/성취도/이탈 위험)

---

## Notes

- TabNet은 현재 실시간 추론 UI에서 제외되어 있으며, 학습/분석 파이프라인에서 사용합니다.
- 분석 탭에서는 `model_name`을 정규화해 모델 중복 표기를 방지합니다.

---

## Docs

운영 문서(로컬 참고용):
- `env/WORKFLOW.md`
- `env/PROJECT_PLAN.md`
- `env/SETUP.md`
