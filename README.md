# OULAD 이탈 예측

OULAD 데이터셋으로 수강 이탈 여부를 예측하는 이진 분류 팀 프로젝트.

자세한 내용은 [PROJECT_PLAN.md](PROJECT_PLAN.md)를 참고하세요.

---

## 팀원 세팅

`setup.sh` 파일을 받아서 실행합니다. 자세한 내용은 [SETUP.md](SETUP.md)를 참고하세요.

---

## 서버 세팅

```bash
# .env 파일 생성
cat > .env << EOF
MYSQL_ROOT_PASSWORD=root1234
MYSQL_DATABASE=dropout
MYSQL_USER=dropout_user
MYSQL_PASSWORD=dropout1234
EOF

docker compose -f docker-compose.server.yml up -d
```

---

## 접속 주소

| 서비스 | 주소 |
|---|---|
| Jupyter | http://localhost:9000 |
| Streamlit | http://localhost:8501 |
| MLflow | http://서버-IP:5001 |

---

## 브랜치 구조

```
main                  ← 발표/제출용 (건드리지 않음)
└── dev               ← 통합용 (완성된 것만 merge)
    ├── model/xgboost        ← 담당 1
    ├── model/logistic       ← 담당 2
    ├── model/random-forest  ← 담당 3
    ├── model/tabnet         ← 담당 4
    ├── analysis/clustering  ← 담당 5
    ├── app/streamlit        ← Phase 3에서 완성
    ├── infra/docker
    ├── infra/mysql
    └── infra/mlflow
```

## 작업 방법

```bash
# 1. 레포 클론
git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN25-2nd-3Team.git
cd SKN25-2nd-3Team

# 2. 본인 브랜치로 이동 (예: 담당 1)
git checkout model/xgboost

# 3. 작업 후 커밋 & 푸시
git add .
git commit -m "feat: XGBoost 학습 및 MLflow 기록 추가"
git push origin model/xgboost

# 4. GitHub에서 PR 생성: 본인브랜치 → dev
```

### 커밋 컨벤션

| 태그 | 용도 |
|---|---|
| `feat:` | 기능 추가 |
| `fix:` | 버그 수정 |
| `refactor:` | 코드 정리 (기능 변경 없음) |
| `chore:` | 설정, Docker, 의존성 |
| `docs:` | 문서, 주석 |
| `data:` | 스키마, MySQL |

```bash
# 예시
feat: XGBoost 학습 및 MLflow 기록 추가
fix: logistic.py StandardScaler 누락 수정
chore: Docker Compose MySQL 포트 수정
data: predictions 테이블 스키마 추가
```

### PR 규칙

| 경로 | 조건 |
|---|---|
| 본인브랜치 → `dev` | 담당자 외 1인 승인 필요 |
| `dev` → `main` | 전원 승인 필요 |

- PR 제목은 커밋 컨벤션 태그로 시작 (예: `feat: XGBoost 학습 추가`)
- 리뷰 없이 직접 `dev`, `main`에 push 금지
