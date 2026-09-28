# 제조 전력 예측: 경진대회 ⑤ 자원 최적화

제6회 K-인공지능 제조데이터 분석 경진대회 ⑤번 과제의 실험 코드입니다. 원문은 **향후 지정 시간구간의 전력 예측, 큰 오차와 피크 조건 분석, 운영 조정안**을 요구합니다. 제공된 CSV에는 별도 평가 파일이나 지정 예측 구간이 없어, 여기서는 **직전 시간 종료 시점에 다음 1시간의 `평균` 전력을 예측**하는 문제로 명시해 실험합니다. 실제 목표 시점과 전력 단위는 공식 데이터 설명이 있으면 재확인해야 합니다.

## 실행

Python 3.10 환경에서:

```bash
python3 -m pip install -r requirements.txt
python3 forecast.py --data okm_augumented_2021.csv --output outputs
python3 audit.py
python3 advanced_forecast.py
python3 deploy.py train
python3 deploy.py predict
python3 -m unittest discover -s tests
```

`outputs/summary.json`에는 데이터 진단, 시간순 검증, 시험 성능, 월별 재학습 및 사후 조건 분석이 들어 있습니다. `outputs/test_predictions.csv`에는 다음 시간 평균 및 15분 최대값 예측, 회귀·분류 피크 경보, 평균값 예측 범위가 들어 있습니다. `outputs/test_first_week.png`는 평균 전력 예측 그림입니다.

## 평가 설계

- 2021년 1~6월 학습, 7월 검증, 8월~9월 14일 시험. 모델과 초매개변수 선택은 4~7월의 확장 학습창 검증 4개 구간에서 합니다.
- 동일 시간의 `15분`·`30분`·`45분`·`60분`, 생산량, 날씨는 미래 실측치이므로 예측 입력으로 사용하지 않습니다. 달력과 과거 관측만 사용합니다.
- `시간`이 손상된 2021년 7월 13일과 15일은 시각을 복원할 근거가 없어 제외합니다. 정확한 지연 특징이 그 간격을 건너지 않도록 빈 시간을 유지합니다.
- 시간 평균 피크는 **1~6월 시간 평균 전력의 95백분위**, 15분 최대값 피크는 **1~6월 시간별 15분 최대값의 95백분위**를 임시 기준으로 삼습니다. 두 목표 모두 미래의 해당 시간 관측값은 학습 정답이나 사후 평가에만 사용합니다.

단계별 변경과 수치는 [STAGE1.md](STAGE1.md), [STAGE2.md](STAGE2.md), [STAGE3.md](STAGE3.md), [STAGE4.md](STAGE4.md), [STAGE5.md](STAGE5.md), [STAGE6.md](STAGE6.md)에 있습니다. 6차에는 데이터 반복성과 월별 성능 개선의 한계를 다시 검증했습니다. 기존 분석은 [ANALYSIS.md](ANALYSIS.md), 과제 원문 대비 한계와 제출 항목은 [ROADMAP.md](ROADMAP.md)에 남겨두었습니다.

[7차 실험](STAGE7.md)은 6개 후보를 같은 조건에서 비교한 뒤 기존 모델을 유지했습니다. 후보별 검증 예측과 선택 결과는 `outputs/stage7/`에 있습니다. 새 후보를 추가한 만큼 점수가 개선됐다고 해석하지 않습니다.

[8차 추론](STAGE8.md)은 정답이 없는 다음 한 시간을 예측합니다. 저장한 모델은 `artifacts/latest.joblib`, 학습·입력 기록은 `artifacts/latest.json`, 예측은 `outputs/stage8/next_prediction.json`에 생성됩니다. 모델 바이너리는 위 명령으로 재생성합니다. 과거 재현을 위한 모델 학습 종료 시점과 `--last-observed-hour` 사용법은 8차 문서에 있습니다.

현재 평균 전력 평가 MAE는 **5.067**이며, 개발 중 이미 확인한 8~9월에 대한 월별 재학습 결과입니다. 새로운 독립 시험 점수가 아닙니다. 피크 직접 분류는 같은 기간 미탐 7시간, 오경보 126시간으로, 현장 비용에 따른 선택이 필요합니다.

[9차 특징 실험](STAGE9.md)은 `python3 research.py --stage 9`로 실행합니다. 4~6월로 선택하고 7월을 별도로 진단하며, 선택 기준과 다음 실험 계획은 [EXPERIMENTS.md](EXPERIMENTS.md)에 기록했습니다.

[10차 알고리즘 비교](STAGE10.md)는 `python3 research.py --stage 10`으로 재현합니다. 선택된 ExtraTrees의 8~9월 MAE는 4.904입니다. 8차 저장 모델은 아직 이 후보로 교체하지 않았으며 후속 검증 후 연결합니다.

[11차 결합 실험](STAGE11.md)은 `python3 research.py --stage 11`로 재현합니다. 검증은 ExtraTrees 단독을 선택했으며, 결합에 따른 점수 향상은 없었습니다.
