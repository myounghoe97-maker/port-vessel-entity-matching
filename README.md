# 항만 선박 Entity Matching

입출항 기록을 선박 카탈로그의 동일 선박과 연결하고, 판단 근거가 부족한 기록을 검토 대상으로 분리하는 프로젝트입니다. 데이터 정규화부터 후보 검색, 순위 계산, Confidence Gate, Streamlit CSV 서비스까지 연결했습니다.

이 저장소는 **실제 서비스 코드의 공개용 사본**입니다. 원본 데이터·학습 결과·모델 바이너리는 포함하지 않습니다. 설치 후 합성 데이터 smoke test는 실행할 수 있지만, 비공개 자산 없이 실제 식별 서비스를 재현할 수는 없습니다.

## 문제 정의와 Entity Matching을 선택한 이유

입출항 기록의 선명만으로 카탈로그의 고유 선박 ID를 안정적으로 찾기 어렵습니다. 한글·영문 표기 차이, 공백과 구분자, 유사한 이름, 누락된 GT(총톤수), 선종·국적의 표현 차이가 비교를 어렵게 합니다. 이름 완전 일치만 사용하는 대신, 각 입력과 검색된 후보의 여러 속성을 비교해 동일 개체 가능성을 순위화하는 Entity Matching으로 구성했습니다.

## 전체 파이프라인

```mermaid
flowchart LR
    A[입출항 CSV] --> B[입력 검증·정규화]
    B --> C[R_B100_V1 후보 검색]
    C --> D[후보별 Feature]
    D --> E[M6 점수·V4.2 순위 보정]
    E --> F[Confidence Gate·검토 정책]
    F --> G[자동 식별 / 검토 필요]
    G --> H[결과·검토 보고서 다운로드]
```

### 데이터 정규화

- 원본 값은 `*_raw`로 보존하고 비교용 값을 별도로 만듭니다.
- 선명: NFKC, 대문자화, 공백 및 지정된 구분자 제거. 번역이나 임의 사전 치환은 하지 않습니다.
- GT: 숫자 파싱과 누락·오류 상태를 분리합니다.
- 선종·국적: 고정 매핑 테이블을 사용하며, 처음 보는 값은 `UNSEEN_RAW`로 구분합니다. 매핑 테이블은 비공개입니다.

### 후보 검색

`core/retrieval.py`의 `R_B100_V1`은 정규화 이름 완전 일치 후보가 있으면 ID 순으로 최대 100개를 반환합니다. 없으면 문자 trigram 사전검색과 SequenceMatcher를 이용한 Fuzzy Name Top25, GT·선종·국적 신호를 사용하는 G3 Top75를 합치고 중복을 제거합니다. 최종 후보는 최대 100개입니다.

### Ranking

`core/ranker.py`에서 확인되는 M6 모델 정의는 scikit-learn `LogisticRegression`입니다. 숫자 피처에는 결측치 대체·표준화, 범주형 피처에는 결측치 대체·One-hot encoding을 적용합니다. 이름 완전 일치·유사도, GT 차이, Korean-Latin Rule Support, 선종·국적 비교 상태를 사용합니다.

서비스는 모델을 다시 학습하지 않고 SHA256과 피처 메타데이터를 검사한 고정 모델을 로드합니다. `predict_proba` 점수로 정렬한 다음, 기존 V4.2 KL 충돌 규칙에 해당하는 후보를 뒤로 배치합니다. **V4.2 보정은 실제 순위를 바꾸는 기존 backend 규칙**이며, 이 공개 작업에서 새로 만든 규칙이 아닙니다. 공개 사본은 기존 점수·순위·판정 코드를 변경하지 않았습니다.

### Confidence Gate와 검토 필요 분류

V1 gate는 후보 1개 또는 1·2위 점수 차이 0.20 이상이면 `AUTO_MATCH`, 나머지는 `REVIEW_REQUIRED`로 분류합니다. 현재 서비스는 여기에 한글·영문 교차 표기, 이름·국적 충돌 및 근접 유사도 검토 정책을 적용합니다. 이 검토 정책은 자동 판정을 검토로 전환할 수 있지만 후보 점수·순위를 수정하지 않습니다.

이미 검토 대상으로 분류된 입력에서 GT·국적 누락 때문에 상위 후보를 구분하기 어려운 경우, 추가 확인할 정보를 표시합니다. 이 정보 부족 표시는 presentation only이며 backend rank·score·decision을 유지합니다. 처리 불가 입력과 후보 없음은 각각 `NOT_PROCESSED`, `NO_CANDIDATE`로 구분합니다.

## 검증 방법과 결과 해석

원본 연구 기록의 STEP 19 Frozen VESSEL_GROUP TEST는 2,344건입니다. 해당 **역사적 기준선**의 Retrieval Recall은 97.6109%, End-to-End Top1은 90.7423%입니다. STEP 20 V1 gate 평가에서는 자동 처리 비율 84.8123%, 자동 처리 집합 내 precision 99.7988%를 기록했습니다. 자동 처리 집합 precision은 전체 입력의 정확도가 아닙니다.

위 수치는 후속 V4.2 및 서비스 검토 정책 전체의 최신 성능을 의미하지 않습니다. Frozen Test는 완전히 새로운 외부 데이터 평가와 구분해야 합니다. 이 공개 작업에서는 외부 일반화 성능을 새로 측정하거나 주장하지 않습니다. 버전별 근거와 검증 한계는 [evaluation.md](docs/evaluation.md)를 확인하세요.

공개 코드 검증은 Python syntax, 모듈 import, 합성 데이터 기반 입력 검증·검색·피처·gate·순위 보정·배치 smoke test로 구성합니다. 복사한 서비스와 core 모듈은 원본과 바이트 단위 SHA256이 일치하며, 정규화 추출 함수는 원본 AST와 일치하는지 확인했습니다.

## 기술 스택

Python 3.12, NumPy, pandas, scikit-learn, joblib, Streamlit. 의존 버전은 원본 `requirements-service.txt`를 그대로 옮긴 `requirements.txt`에 고정했습니다. 테스트는 표준 라이브러리 unittest를 사용합니다.

## 프로젝트 구조

```text
core/
  validation.py       입력 검증·원본 값 보존
  normalization.py    고정 정규화·매핑 연결
  retrieval.py        R_B100_V1 후보 검색
  features.py         이름 유사도·KL 피처
  ranker.py           M6 모델 정의
  confidence.py       고정 V1 gate
  review_safety.py    기존 검토 정책
  batch_pipeline.py   서비스 실행 파이프라인
service/app.py         CSV 업로드·결과 표시·보고서
06_build_pair_features.py  원본 normalize_name 함수만 추출
tests/test_public_smoke.py 합성 데이터 테스트
docs/                  구조·검증·공개 범위
examples/README.md     입력 스키마
```

## 실행 방법

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
```

위 테스트는 비공개 자산 없이 실행합니다. 모델을 새로 학습하거나 성능 수치를 재현하는 명령이 아닙니다.

실제 서비스는 권한이 있는 원본 자산을 [명세](docs/private-assets.md)에 따라 로컬에 준비한 경우에만 실행할 수 있습니다.

```powershell
python -m streamlit run service/app.py
```

비공개 자산이 없으면 매처 초기화가 실패하는 것이 현재 코드의 동작입니다. 공개용 가짜 모델이나 대체 점수는 넣지 않았습니다. 운영 서비스는 별도 비공개 배포 환경에 있으며 이 저장소에서 배포를 변경하지 않습니다.

## 보안·데이터 공개 범위

핵심 코드·문서·합성 테스트만 공개합니다. 원본 및 학습 CSV, 선박 카탈로그, 매핑·학습된 규칙, 모델 바이너리, 중간 결과, 개인정보, 인증정보, 개인 절대경로, 비공개 저장소 정보는 제외했습니다. `.gitignore`는 데이터·결과·모델·환경설정 secret을 차단합니다. [공개 범위 및 검증 기록](docs/publication.md)에 파일 분류와 확인 범위를 기록했습니다.
