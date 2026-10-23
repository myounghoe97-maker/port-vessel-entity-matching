# 입력 스키마

필수 컬럼: `vessel_name`, `gross_tonnage`, `vessel_type`, `flag_name`.
서비스에서 생성하는 예약 컬럼을 입력에 추가하지 마세요. 최대 5,000행입니다.

다음은 실제 선박과 무관한 합성 예시입니다. CSV 파일은 저장소에 포함하지 않습니다.

```python
import pandas as pd
example = pd.DataFrame([{
    "vessel_name": "SYNTHETIC ALPHA",
    "gross_tonnage": "1000",
    "vessel_type": "SYNTHETIC TYPE",
    "flag_name": "SYNTHETIC FLAG",
}])
```

이름과 GT가 모두 없으면 처리 불가로 분류합니다. 파싱할 수 없는 GT 등은 입력 검증에서 구분합니다. 원본 선종·국적 매핑이 없으므로 위 예시에 실제 식별 결과를 부여하지 않습니다.
