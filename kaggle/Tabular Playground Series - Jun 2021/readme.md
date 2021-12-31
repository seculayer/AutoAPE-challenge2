# Tabular Playground Series - Jun 2021
## 결과
### 요약정보
- 도전기관: 한양대학교
- 도전자: 김홍식
- 최종스코어: 1.74507
- 제출일자: 2021-12-30
- 총 참여 팀수: 1171
- 순위 및 비율: 41.42%
### 결과화면
![leaderboard15](./img/leaderboard15.png)
## 사용한 방법 & 알고리즘
- ensemble of CatBoost model with parameters:
model = CatBoostClassifier(iterations=iterations,
                               learning_rate=learning_rate,
                               depth=depth, # 10
                               loss_function='MultiClass',
                               min_data_in_leaf=11,
                               reg_lambda=53.3,
                               random_strength=32,
                               verbose=10)
model0 = getCatBoostModel(1500, 7, 0.032) # catboost classifier
model1 = getCatBoostModel(2000, 7, 0.032) # catboost classifier
model2 = getCatBoostModel(2667, 7, 0.032) # catboost classifier
model3 = getCatBoostModel(1500, 8, 0.032) # catboost classifier
model4 = getCatBoostModel(2000, 8, 0.032) # catboost classifier
model5 = getCatBoostModel(2667, 8, 0.032) # catboost classifier
model6 = getCatBoostModel(1500, 7, 0.025) # catboost classifier
model7 = getCatBoostModel(2000, 7, 0.025) # catboost classifier
model8 = getCatBoostModel(2667, 7, 0.025) # catboost classifier
## 코드
https://github.com/WannaBeSuperteur/2020/tree/master/AI/kaggle/2021_09_TPS_Jun_2021