# Tabular Playground Series - Jan 2021
## 결과
### 요약정보
- 도전기관: 한양대학교
- 도전자: 김홍식
- 최종스코어: 0.69683
- 제출일자: 2021-02-26
- 총 참여 팀수: 1728
- 순위 및 비율: 22.45%
### 결과화면
![leaderboard00](./img/leaderboard00.png)
## 사용한 방법 & 알고리즘
- lightGBM with parameters:
params = {'learning_rate': 0.01,
              'max_depth': -1,
              'boosting': 'gbdt',
              'objective': 'regression',
              'metric': 'rmse',
              'is_training_metric': True,
              'num_leaves': 256,
              'feature_fraction': 0.5,
              'bagging_fraction': 0.6,
              'bagging_freq': 3,
              'seed': None}
## 코드
https://github.com/WannaBeSuperteur/2020/tree/master/AI/kaggle/2021_02/tabular_playground_jan_2021