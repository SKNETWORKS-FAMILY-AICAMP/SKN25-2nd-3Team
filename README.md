<div align="center">

# 🎓 OULAD Dropout Analytics

**Open University Learning Analytics Dataset 기반<br>학습 이탈 예측 및 군집 분석 대시보드**

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://mysql.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189B48?style=for-the-badge&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

</div>

---

## 📌 프로젝트 소개

> **OULAD(Open University Learning Analytics Dataset)**을 기반으로 학습자의 이탈(Dropout)을 예측하고, 군집 분석을 통해 학습자 유형을 분류하는 인터랙티브 대시보드 프로젝트입니다.

- 🔮 **이탈 예측**: Logistic Regression, Random Forest, XGBoost, CatBoost 모델을 활용한 실시간 추론
- 📊 **분석 대시보드**: 모델별 성능 비교, 예측 분포, 그룹 비교 시각화
- 🧩 **군집 분석**: KMeans 기반 학습자 유형 분류 및 군집 프로파일 해석
- 📦 **실험 추적**: MLflow 기반 실험 관리 + MySQL 예측 결과 저장

---

## 🏗️ 시스템 아키텍처

<div align="center">

![Image](https://github.com/user-attachments/assets/bad6532c-14e4-4cb2-8d35-d5d946af900f)

</div>

| 구성 요소 | 포트 | 역할 |
|---|---|---|
| 🖥️ Streamlit | :8501 | 이탈 예측 & 군집 분석 대시보드 |
| 🧪 MLflow | :5001 | 실험 추적 & 모델 레지스트리 |
| 📓 Jupyter | :9000 | EDA / 탐색 노트북 |
| 🗄️ MySQL | :3307 | OULAD 데이터 & 예측 결과 저장 |
| 🔒 Tailscale | — | VPN Mesh 네트워크 (포트포워딩 대체) |

---

## 🗂️ 레포지토리 구조

```
SKN25-2nd-3Team/
│
├── 📁 src/
│   ├── app.py                         # Streamlit 메인 앱
│   ├── common_data.py                 # 공통 데이터 로더
│   ├── 📁 models/
│   │   ├── logistic.py                # 로지스틱 회귀
│   │   ├── random_forest.py           # 랜덤 포레스트
│   │   ├── xgboost.py                 # XGBoost
│   │   ├── catboost.py                # CatBoost
│   │   ├── tabnet.py                  # TabNet (PyTorch)
│   │   └── clustering.py             # KMeans 군집 분석
│   └── 📁 pages/
│       ├── 01_dropout_prediction.py  # 이탈 예측 페이지
│       └── 02_clustering_analysis.py # 군집 분석 페이지
│
├── 📁 scripts/
│   └── init_db.py                     # DB 초기화 스크립트
│
├── 📁 docker_env/                     # Docker 환경 설정
├── 📁 outputs/                        # 분석 결과 출력물
├── docker-compose.yml
├── docker-compose.server.yml
└── README.md
```

---

## 🗄️ ERD (데이터베이스 구조)

<div align="center">

![Image](https://github.com/user-attachments/assets/96bb63bf-dcf2-43a3-bd9d-e0588a5a447c)

</div>

---

## ⚡ 빠른 시작 (Quick Start)

### 1️⃣ 환경 변수 설정

`.env` 파일을 생성하고 아래 변수를 설정하세요:

```env
DB_USER=your_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=3307
DB_NAME=oulad
MLFLOW_TRACKING_URI=http://localhost:5001
```

### 2️⃣ 서비스 실행

```bash
docker compose up -d
```

### 3️⃣ DB 초기화 (최초 1회)

```bash
python scripts/init_db.py
```

### 4️⃣ 접속

| 서비스 | URL |
|---|---|
| 🖥️ Streamlit 대시보드 | http://localhost:8501 |
| 📓 Jupyter Notebook | http://localhost:9000 |
| 🧪 MLflow UI | http://localhost:5001 |

---

## 🤖 모델 학습 & 로깅

각 모델 스크립트 실행 시 **학습 → MLflow 로깅 → DB 저장**이 자동으로 수행됩니다.

```bash
# 분류 모델
python src/models/logistic.py
python src/models/random_forest.py
python src/models/xgboost.py
python src/models/catboost.py
python src/models/tabnet.py

# 군집 분석
python src/models/clustering.py
```

---

## 🖥️ Streamlit 대시보드 페이지

### 📍 Page 1 — 이탈 예측

| 기능 | 설명 |
|---|---|
| 실시간 추론 | Logistic Regression / Random Forest / XGBoost / CatBoost |
| 입력 모드 | 간소화 입력 / 전체 입력 전환 가능 |
| 저장 기능 | `predictions` 테이블에 예측 결과 실시간 저장 |

### 📍 Page 2 — 군집 분석

| 기능 | 설명 |
|---|---|
| 군집 KPI | 군집별 핵심 지표 요약 |
| 분포 시각화 | 군집 내 특성 분포 차트 |
| 프로파일 비교 | 참여도 / 성취도 / 이탈 위험 해석 카드 |

---

## 🔧 PyTorch (TabNet) 설치 가이드

환경에 따라 아래 명령어로 PyTorch를 설치하세요:

<details>
<summary>🪟 Windows + NVIDIA GPU (CUDA)</summary>

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pytorch-tabnet
```

</details>

<details>
<summary>🍎 Apple Silicon (M1/M2/M3)</summary>

```bash
pip install torch torchvision torchaudio
pip install pytorch-tabnet
```

</details>

> 💡 Docker 컨테이너 환경의 기본 의존성은 `docker_env/requirements.txt`에서 관리됩니다.

---

## 📝 Notes

- ⚠️ **TabNet**은 현재 실시간 추론 UI에서 제외되어 있으며, 학습/분석 파이프라인에서만 사용됩니다.
- 분석 탭에서는 `model_name`을 정규화하여 모델 중복 표기를 방지합니다.

---

## 📄 운영 문서

| 문서 | 경로 |
|---|---|
| 워크플로우 | `env/WORKFLOW.md` |
| 프로젝트 플랜 | `env/PROJECT_PLAN.md` |
| 환경 설정 | `env/SETUP.md` |

---

## 👥 팀원

<div align="center">

| 🧹 김서현 | 🏅 김주희 | 🎩 김찬영 | 🧝 이상민 | 🦉 최원준 |
|:---:|:---:|:---:|:---:|:---:|
| 군집 분석 | XGBoost<br>CatBoost | Logistic<br>Regression | Random<br>Forest | 환경 구축<br>TabNet |

</div>

---

<div align="center">

**SKN25 2기 3팀** · Built with ❤️ by SKNETWORKS-FAMILY-AICAMP

![Python](https://img.shields.io/badge/Python-98.9%25-3776AB?style=flat-square&logo=python&logoColor=white)
![Dockerfile](https://img.shields.io/badge/Dockerfile-1.1%25-2496ED?style=flat-square&logo=docker&logoColor=white)

</div>
