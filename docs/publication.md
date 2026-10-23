# 공개 범위와 검증 기록

검증일: 2026-10-23. Python 3.12 원본 프로젝트 환경에서 실행했습니다.

## 공개 파일 분류

- 원문 복사: 현재 활성 core 9개 모듈(초기화 포함), service 2개 파일. `source-integrity.json`에 SHA256을 기록했습니다.
- 제한 추출: STEP 6의 `normalize_name` 함수와 필요한 import만 공개. 원본 함수 AST와 일치합니다.
- 새 문서 및 테스트: README, 구조·평가·자산 명세·공개 기록, 합성 테스트, 입력 스키마, 의존성, ignore 및 라이선스.
- 원본의 빈 `core/pipeline.py`, backup 파일, automation 기반 `run_project.py`와 `project_config.json`은 공개 실행 경로에서 제외했습니다.
- 지정된 원본에는 `cloud_deploy/`가 없었습니다. 별도의 배포 저장소를 가져오거나 수정하지 않았습니다.

## 제외한 자료

원본·학습 CSV, 카탈로그, 매핑 테이블, 학습된 KL 규칙, 모델 바이너리와 메타데이터, results 및 artifacts, 19GB Hard Negative Pool, 가상환경, 로그·캐시·백업, 원본 Git 이력·설정, 개인정보와 인증정보, 배포 환경 설정, 개인 절대경로가 있는 연구 runner. Hard Negative Pool 내용은 읽지 않았습니다.

## 공개 전 검사

| 항목 | 결과 | 범위 |
|---|---|---|
| Python syntax | PASS | 공개 Python 전체 AST parse |
| core import | PASS | 활성 모듈 8개 및 동적 정규화 로드 |
| Streamlit 초기 화면 | PASS | AppTest 실행, exception 0개, CSV uploader 확인; 실제 식별은 미실행 |
| 합성 smoke | PASS | unittest 7개, production 모델/데이터 미사용 |
| 원본 동일성 | PASS | 복사 파일 바이트 SHA256 및 추출 함수 AST |
| Secret·인증정보 패턴 | PASS | 공개 텍스트 전체 검사 및 활성 코드 내용 확인 |
| 개인 절대경로·비공개 저장소·이메일 | PASS | 공개 텍스트 전체 검사 |
| 원본 CSV·모델·대용량 파일 | PASS | 공개 파일 확장자 및 개별 1MB 상한 검사 |
| README/구현 대조 | PASS | 모델·검색·V4.2·gate·검토 정책·비공개 의존성 확인 |

보안 검사는 공개 허용목록 파일에 대한 패턴 검사와 내용 검토입니다. 비공개 프로젝트 전체 데이터의 포렌식 검사나 외부 보안 인증은 아닙니다. 민감 자산은 검사 결과에 관계없이 복사 대상에서 제외했습니다.

실제 fitted 모델과 데이터가 없어 공개본 단독의 운영 예측 및 전체 성능 재현은 검증하지 않았습니다. 원본 서비스 코드를 그대로 보존했으며 비공개 자산을 숨겨서 대체하는 fallback은 없습니다. 테스트의 고정 점수 stub은 테스트 내에서만 적용합니다.
