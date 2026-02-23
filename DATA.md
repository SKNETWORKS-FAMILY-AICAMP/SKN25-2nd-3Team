# DATA.md — 데이터 구조 및 전처리 기준

## 1. ERD

```
┌──────────────────────────────────────────────┐
│                   students                   │
├──────────────────┬───────────┬───────────────┤
│ 컬럼             │ 타입      │ 비고           │
├──────────────────┼───────────┼───────────────┤
│ id_student       │ INT       │               │
│ code_module      │ VARCHAR   │ 수강 모듈 코드 │
│ code_presentation│ VARCHAR   │ 연도/학기 코드 │
│ gender           │ VARCHAR   │ M / F         │
│ region           │ VARCHAR   │ 13개 지역      │
│ highest_education│ VARCHAR   │ 5개 학력 수준  │
│ imd_band         │ VARCHAR   │ 10분위 소득    │
│ age_band         │ VARCHAR   │ 3개 연령대     │
│ num_of_prev_attempts│ INT    │               │
│ studied_credits  │ INT       │               │
│ disability       │ VARCHAR   │ Y / N         │
│ total_clicks     │ FLOAT     │               │
│ active_days      │ FLOAT     │               │
│ unique_resources │ FLOAT     │               │
│ num_forum        │ FLOAT     │               │
│ num_quiz         │ FLOAT     │               │
│ avg_score        │ FLOAT     │               │
│ num_assess_attempted│ FLOAT  │               │
│ total_weight     │ FLOAT     │               │
│ module_presentation_length│ INT│             │
│ dropout          │ TINYINT   │ 0/1 (타겟)    │
└──────────────────┴───────────┴───────────────┘
                    │ id_student
           ┌────────┴────────┐
           ▼                 ▼
┌────────────────────┐  ┌──────────────────┐
│     predictions    │  │     clusters     │
├────────┬───────────┤  ├────────┬─────────┤
│id_student  │ INT   │  │id_student  │ INT  │
│model_name  │VARCHAR│  │cluster_id  │ INT  │
│predicted   │TINYINT│  │dropout_rate│FLOAT │
│probability │ FLOAT │  └────────────┴──────┘
│run_id      │VARCHAR│
└────────────┴───────┘
       │ run_id
       ▼
  MLflow runs
```

### 관계 요약
| 관계 | 설명 |
|------|------|
| `students` → `predictions` | 1:N (한 학생, 여러 모델 예측 결과) |
| `students` → `clusters` | 1:1 (한 학생, 클러스터 1개 배정) |
| `predictions.run_id` → MLflow | 실험 추적 연결 |

---

## 2. 타겟 정의

| 값 | 의미 | 원본 final_result |
|----|------|-------------------|
| 0  | 이탈 아님 | Pass, Fail, Distinction |
| 1  | 이탈 | Withdrawn |

- 이탈자 수: **10,156명** (31.2%)
- 비이탈자 수: **22,437명** (68.8%)

---

## 3. 기초 전처리 기준 (df_base)

DB에서 불러온 raw 데이터에 대해 모든 모델 공통으로 적용하는 전처리.
인코딩은 모델별 Pipeline에서 처리.

### 3-1. 드롭 컬럼

| 컬럼 | 이유 |
|------|------|
| `id_student` | 식별자, 학습에 불필요 |

### 3-2. Binary 인코딩 (모든 모델 공통)

| 컬럼 | 변환 |
|------|------|
| `gender` | M → 1, F → 0 |
| `disability` | Y → 1, N → 0 |

### 3-3. 모델별 분기

| 컬럼 | 트리 계열 (RF, XGB 등) | 선형 계열 (LR, SVM 등) |
|------|----------------------|----------------------|
| `imd_band` | OrdinalEncoder | OrdinalEncoder |
| `age_band` | OrdinalEncoder | OrdinalEncoder |
| `highest_education` | OrdinalEncoder | OrdinalEncoder |
| `region` | LabelEncoder | OneHotEncoder |
| `code_module` | LabelEncoder | OneHotEncoder |
| `code_presentation` | LabelEncoder | OneHotEncoder |
| 수치형 전체 | 스케일링 불필요 | StandardScaler |

### 3-4. Ordinal 순서 정의

**imd_band** (소득 분위, 낮을수록 취약)
```
0-10% < 10-20% < 20-30% < 30-40% < 40-50%
< 50-60% < 60-70% < 70-80% < 80-90% < 90-100%
```

**age_band**
```
0-35 < 35-55 < 55<=
```

**highest_education**
```
No Formal Quals < Lower Than A Level < A Level or Equivalent
< HE Qualification < Post Graduate Qualification
```
