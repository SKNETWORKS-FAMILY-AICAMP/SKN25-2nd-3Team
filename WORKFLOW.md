# 팀 작업 가이드 (A to Z)

> 비전공자도 따라할 수 있도록 처음부터 끝까지 정리한 작업 흐름입니다.

---

## 목차

1. [전체 구조 이해하기](#1-전체-구조-이해하기)
2. [처음 한 번만 하는 것들](#2-처음-한-번만-하는-것들)
3. [매일 작업 흐름](#3-매일-작업-흐름)
4. [모델 작업 예시 코드](#4-모델-작업-예시-코드)
5. [작업 저장하기 (커밋 & 푸쉬)](#5-작업-저장하기-커밋--푸쉬)
6. [PR 만들기](#6-pr-만들기)
7. [자주 쓰는 명령어 모음](#7-자주-쓰는-명령어-모음)
8. [문제 해결](#8-문제-해결)

---

## 1. 전체 구조 이해하기

### 내 컴퓨터에서 실행되는 것

```
내 컴퓨터 (localhost)
├── Jupyter  → http://localhost:9000  (코드 작성, 실험 공간)
└── Streamlit → http://localhost:8501  (최종 발표 앱 - Phase 3에서 씀)
```

### 서버(담당5 컴퓨터)에서 실행되는 것

```
서버 (100.113.167.80)
├── MySQL  → 100.113.167.80:3307  (팀 공유 데이터베이스)
└── MLflow → http://100.113.167.80:5001  (실험 결과 모아보는 곳)
```

### 전체 흐름 그림

```
내 Jupyter (localhost:9000)
    │
    ├── MySQL에서 데이터 읽기  ──────→ 서버 MySQL :3307
    │
    ├── 모델 학습
    │
    ├── mlflow.log_*() 호출  ────────→ 서버 MLflow :5001
    │                                      └── 모든 팀원 실험 기록됨
    └── 코드 커밋 & 푸쉬  ──────────→ GitHub (본인 브랜치)
```

> **핵심:** 내 컴퓨터는 코드 실행 공간, 서버는 데이터/실험결과 저장 공간

---

## 2. 처음 한 번만 하는 것들

### 2-1. 환경 설치

`setup.sh`를 받아서 실행했다면 이미 완료입니다. → [SETUP.md](SETUP.md) 참고

### 2-2. 내 브랜치 만들기

WSL 터미널(또는 Mac 터미널)을 열고:

```bash
cd ~/workspace/oulad

# 본인 브랜치로 이동 (없으면 자동 생성)
git checkout -b model/tabnet      # 담당 4 예시

# GitHub에 브랜치 올리기 (최초 1회)
git push -u origin model/tabnet
```

**브랜치 이름표:**

| 담당 | 브랜치 이름 |
|---|---|
| 담당 1 | `model/xgboost` |
| 담당 2 | `model/logistic` |
| 담당 3 | `model/random-forest` |
| 담당 4 | `model/tabnet` |
| 담당 5 | `analysis/clustering` |

### 2-3. 컨테이너 실행

```bash
cd ~/workspace/oulad
docker compose up -d
```

이후 브라우저에서 `http://localhost:9000` 열기

---

## 3. 매일 작업 흐름

### 순서

```
① Tailscale 연결 확인
② 컨테이너 실행 (이미 켜져 있으면 생략)
③ Jupyter 접속 → http://localhost:9000
④ 코드 작성 & 실험
⑤ 커밋 & 푸쉬
```

### ① Tailscale 연결 확인

서버(MySQL, MLflow)에 접근하려면 Tailscale이 켜져 있어야 합니다.

```bash
# 연결 상태 확인
tailscale status

# 연결이 안 되어 있으면
sudo tailscale up
```

### ② 컨테이너 실행

```bash
cd ~/workspace/oulad
docker compose up -d

# 실행 확인
docker ps
```

`jupyter`, `streamlit` 컨테이너가 보이면 OK

### ③ Jupyter 접속

브라우저에서 `http://localhost:9000` 접속

처음 접속 시 토큰 입력 화면이 나오면 터미널에서:
```bash
docker logs oulad-jupyter-1 2>&1 | grep token
```
출력된 URL의 `token=...` 부분 복사해서 입력

### ④ 코드 작성

Jupyter에서 `notebooks/` 폴더에 새 노트북 만들어서 작업합니다.

---

## 4. 모델 작업 예시 코드

### 데이터 불러오기

```python
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv("/home/ict/work/.env")  # 컨테이너 내부 경로

engine = create_engine(
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

df = pd.read_sql("SELECT * FROM students", engine)
print(df.shape)  # (32593, 21)
df.head()
```

### MLflow로 실험 기록하기

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 데이터 준비
X = df.drop("dropout", axis=1)
y = df["dropout"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MLflow 실험 시작
# - MLFLOW_TRACKING_URI가 .env에 설정되어 있어 서버로 자동 전송됨
mlflow.set_experiment("tabnet-dropout-prediction")  # 실험 이름 (본인 모델명으로)

with mlflow.start_run(run_name="run-01"):

    # 하이퍼파라미터 기록
    params = {"n_estimators": 100, "max_depth": 5}
    mlflow.log_params(params)

    # 모델 학습
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 평가 지표 기록
    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # 모델 저장
    mlflow.sklearn.log_model(model, "model")

    print("실험 기록 완료! http://100.113.167.80:5001 에서 확인하세요")
```

> **MLflow UI** `http://100.113.167.80:5001` 에서 모든 팀원 실험 결과를 비교할 수 있습니다.

### 예측 결과 MySQL에 저장하기

```python
# 최종 예측 결과를 DB에 저장
predictions_df = pd.DataFrame({
    "student_id": X_test.index,
    "model_name": "tabnet",
    "predicted_dropout": y_pred,
    "probability": model.predict_proba(X_test)[:, 1]
})

predictions_df.to_sql(
    "predictions",
    engine,
    if_exists="append",  # 기존 데이터 유지하고 추가
    index=False
)
print(f"{len(predictions_df)}건 저장 완료")
```

---

## 5. 작업 저장하기 (커밋 & 푸쉬)

> Git은 "작업 내용을 저장하고 GitHub에 올리는 도구"입니다.
> 커밋 = 저장, 푸쉬 = 업로드라고 생각하면 됩니다.

### WSL 터미널에서

```bash
# 1. 프로젝트 폴더로 이동
cd ~/workspace/oulad

# 2. 현재 브랜치 확인 (본인 브랜치인지 꼭 확인!)
git branch
# * model/tabnet  ← 이렇게 본인 브랜치가 표시되어야 함

# 3. 변경된 파일 확인
git status

# 4. 저장할 파일 선택 (노트북 파일 예시)
git add notebooks/tabnet.ipynb

# 5. 커밋 (저장 메시지 작성)
git commit -m "feat: TabNet 초기 모델 학습 추가"

# 6. GitHub에 업로드
git push origin model/tabnet
```

### 커밋 메시지 규칙

| 태그 | 언제 쓰나 | 예시 |
|---|---|---|
| `feat:` | 새 기능/코드 추가 | `feat: TabNet 학습 코드 추가` |
| `fix:` | 버그 수정 | `fix: 전처리 오류 수정` |
| `refactor:` | 코드 정리 (기능 변화 없음) | `refactor: 함수 분리` |
| `docs:` | 문서, 주석 | `docs: 코드 설명 추가` |

### 주의사항

```bash
# ❌ 절대 하지 말 것
git push origin dev    # dev 브랜치에 직접 푸쉬
git push origin main   # main 브랜치에 직접 푸쉬

# ✅ 항상 본인 브랜치에만 푸쉬
git push origin model/tabnet
```

---

## 6. PR 만들기

> PR(Pull Request) = "내 작업을 팀 공용 브랜치(dev)에 합쳐달라는 요청"

### 순서

1. GitHub 레포지토리 접속
2. 상단에 "Compare & pull request" 버튼 클릭 (본인 브랜치 푸쉬 후 자동으로 뜸)
3. 제목과 설명 작성:
   - 제목: `feat: TabNet 모델 학습 및 MLflow 기록 추가`
   - 설명: 어떤 작업을 했는지 간략히
4. Reviewers에 팀원 1명 이상 지정
5. "Create pull request" 클릭

### PR 방향

```
model/tabnet  →  dev  (본인브랜치 → dev, 팀원 1명 이상 승인 필요)
dev           →  main (전원 승인 필요, 발표 직전에만)
```

---

## 7. 자주 쓰는 명령어 모음

### Git

```bash
git branch              # 현재 브랜치 확인
git checkout model/tabnet  # 브랜치 이동
git status              # 변경된 파일 목록
git log --oneline -5    # 최근 커밋 5개 보기
git pull origin dev     # dev 최신 내용 가져오기
```

### Docker

```bash
docker compose up -d    # 컨테이너 시작
docker compose down     # 컨테이너 종료
docker ps               # 실행 중인 컨테이너 목록
docker logs oulad-jupyter-1  # Jupyter 로그 확인
```

### Tailscale

```bash
tailscale status        # 연결 상태 확인
sudo tailscale up       # 연결
sudo tailscale down     # 연결 해제
```

---

## 8. 문제 해결

**Jupyter 토큰을 모르겠어요**
```bash
docker logs oulad-jupyter-1 2>&1 | grep token
```

**MySQL 연결이 안 돼요**
```bash
# Tailscale 먼저 확인
tailscale status
sudo tailscale up
```

**내 브랜치가 아닌 다른 브랜치에서 작업했어요**
```bash
# 아직 커밋 전이라면
git stash                     # 변경사항 임시 저장
git checkout model/tabnet     # 내 브랜치로 이동
git stash pop                 # 변경사항 복구
```

**git push 할 때 인증을 요구해요**

GitHub Personal Access Token(PAT)이 필요합니다:
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token" → `repo` 권한 체크 → 생성
3. 비밀번호 대신 토큰 붙여넣기

**컨테이너가 안 켜져요 (docker: permission denied)**
```bash
# WSL 터미널을 완전히 닫고 다시 열기
# 그래도 안 되면
sudo usermod -aG docker $USER
# 터미널 재시작
```
