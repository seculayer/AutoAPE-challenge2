# prudential life insurance assessment

## 결과

### 요약정보

- 도전기관 : 시큐레이어
- 도전자 : 최지혁
- 최종스코어 : 0.66788
- 제출일자 : 2021-02-12
- 총 참여 팀 수 : 2618
- 순위 및 비율 : 774(29.56%)

### 결과화면

![leaderboard](./img/leaderboard.png)

## 사용한 방법 & 알고리즘
1. 대회설명
 보험회사인 프루덴셜에서 진행하는 대회이다. 고객을 심사하는 시간을 단축시키기 위한 목적이다. csv파일로 고객에 대한 정보를 받아 반응(1~8, int형)을 예측하는 형식이다.

2. 데이터 설명
 총 훈련 데이터: 59381 rows x 126 columns 
  int형:   113 columns
  float형: 014 columns
  str형:    001 column
 훈련용 Y값과 테스트용 예측값은 모두 int형이다.

3. 알고리즘 설명
 str형 전처리 후 factorize로 인코딩 후 새 피쳐로 추가. kfold와 gridsearch로 조정된 값을 xgboost로 모델 제작.


## 코드

['./main_prudential_life_insurance_assessment.py'](./main_prudential_life_insurance_assessment.py)

## 참고 자료

- 
