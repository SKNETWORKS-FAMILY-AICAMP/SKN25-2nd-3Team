# Configuration file for notebook.

c = get_config()

# 로컬 개발 환경용 — 인증 비활성화 (포트는 localhost에만 바인딩됨)
c.ServerApp.token = ""
c.ServerApp.password = ""
