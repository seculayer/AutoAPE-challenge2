# Yelp Recruiting Competition
## 결과
### 요약정보
- 도전기관: 한양대학교
- 도전자: 김홍식
- 최종스코어: 0.52940
- 제출일자: 2021-05-18
- 총 참여 팀수: 350
- 순위 및 비율: 35.71%
### 결과화면
![leaderboard09](./img/leaderboard09.png)
## 사용한 방법 & 알고리즘
- DataBase Join
- lightGBM with parameters:
params = {'resample': None,
              'boosting': 'gbdt',
              'num_leaves': 100 + count,
              'min_child_samples': 32 + count,
              'max_depth': 16,
              'max_delta_step': 8,
              'reg_alpha': 0.5,
              'reg_lambda': 8.0,
              'colsample_bytree': 0.25,
              'cat_smooth': 80 + count,
              'cat_l2': 12,
              'learning_rate': 0.005, # modified
              'metric': 'rmse'}
## 코드
https://github.com/WannaBeSuperteur/2020/tree/master/AI/kaggle/2021_04_Yelp_recruiting