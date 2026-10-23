# 코드에서 서비스까지

`service/app.py`는 CSV를 읽고 필수 컬럼과 최대 5,000행 제한을 확인합니다. `BatchMatcher.process_dataframe`을 호출하고 결과를 session_state에 보존해 다운로드 때 식별을 반복하지 않습니다. 결과 표시와 HTML 보고서는 원본 backend 상태를 사용자용 한국어로 표현합니다.

`core/validation.py` → `normalization.py` → `retrieval.py` → `batch_pipeline._add_base_pair_features` → `features.add_m6_features` → 고정 모델의 `predict_proba` → `_apply_v42_kl_safety` → V1 gate → `review_safety` → 정보 부족 표시 순서입니다.

검색은 정답 ID를 사용하지 않습니다. 학습된 모델 파라미터와 규칙은 공개하지 않으며, 공개 모델 정의만으로 기존 모델을 재현했다고 주장하지 않습니다. 동률 처리에는 candidate ID를 사용합니다.

V4.2 KL 보정은 KL 일치·GT 비누락·GT 차이 양수·국적 불일치가 동시에 성립할 때 후보를 뒤로 배치하며 삭제하지 않습니다. 이후의 교차 문자 및 이름/국적 검토 정책은 판정만 검토로 바꿀 수 있습니다. 정보 부족 표시는 이미 검토 판정된 행에만 추가됩니다.

빈 원본 `core/pipeline.py` 대신 실제 실행 경로인 `core/batch_pipeline.py`를 공개했습니다. 원본 모듈 경로를 유지해 리팩터링에 따른 동작 변경을 피했습니다.
