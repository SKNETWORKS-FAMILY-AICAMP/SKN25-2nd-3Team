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
| MLflow | http://서버-IP:5000 |
