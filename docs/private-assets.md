# 실제 서비스의 비공개 의존 자산

아래 상대경로는 코드가 요구하는 인터페이스입니다. 파일 자체는 제공하지 않으며 Git 추적 대상도 아닙니다. 임의로 만든 파일이나 다른 모델을 넣으면 원본 동작이 재현되지 않습니다. 신뢰하는 로컬 모델 파일만 사용해야 합니다(joblib 역직렬화).

| 로컬 상대경로 | 역할 |
|---|---|
| results_step4b2/catalog_with_broad_family.csv | 원본 카탈로그, vessel_id/name/GT/type/flag |
| results_step4b2/pilot_vessel_type_mapping.csv | source, raw_vessel_type, broad_family, mapping_status |
| results_step4a_verified/flag_canonical_mapping.csv | source, raw_flag, canonical_iso3, mapping_status |
| results_step15/step15_13f_frozen_rule_registry.csv | 23개 FROZEN KL 규칙, rule_id/position/hangul_part/latin_part |
| results_step27/m6_final_frozen_service.joblib | 실제 고정 fitted 모델 |
| results_step27/m6_final_frozen_service_metadata.json | artifact_sha256, model_features, frozen_ranker 검증 |

서비스 초기화는 모델 해시, 피처 목록, ranker 이름을 확인합니다. 실제 자산이 없는 공개 clone은 단위 smoke test와 코드 검토를 위한 배포물입니다. 과거 연구 스크립트 전체와 학습 runner는 포함하지 않습니다.
