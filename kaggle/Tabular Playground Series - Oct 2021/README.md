## Tabular Playground Series - Oct 2021

------------

### 결과

----------------

### 요약정보

* 도전기관 : 시큐레이어
* 도전자 : 왕승재
* 최종스코어 : 0.85642
* 제출일자 : 2021-10-19
* 총 참여 팀 수 : 804
* 순위 및 비율 : 149 (18%)

### 결과화면

![결과](screanshot/score.png)

![결과](screanshot/leaderboard.png)

----------

### 사용한 방법 & 알고리즘

* Stacked Meta-Learner (Ensemble)
  * Stacking은 비효율적(학습시간)이지만 성능면에서 좋다.
  * 각 fold별로 여러개 모델을 만들고 하나의 fold(학습, 검증)에 대해 예측을 하고 예측 값으로 새로운 column을 추가한다. (5 fold 반복)
  * 기존의 학습 데이터에 각 모델별 prediction 값을 추가해서 학습, 검증 데이터를 만든다.
  * 이 새로운 학습 데이터에 대해 다시 한번 학습시키고 새로운 검증 데이터에서 예측을 진행한다.
  * 본 대회에서는 XGBoost, LightGMB, CatBoost를 기반으로 Ensemble 했다.
  * 최종 Meta-Learner로는 Logistic Regression Model을 사용하여 예측을 진행했다.
  * 200 rounds내에 점수 향상이 없으면 Early Stop을 걸어서 학습 시간을 줄였다.

<img src="screanshot/model1.png" alt="model1" style="zoom:50%;" />

<img src="screanshot/model2.png" alt="model2" style="zoom: 67%;" />

* XGBoost
  * XGBoost는 Gradient Boosting 개념을 의사결정 나무에 도입한 알고리즘이다.
  * 데이터별 오류를 다음 Round 학습에 반영 시킨다는 측면에서 기존 Gradient Boosting과는 큰 차이가 없습니다.
  * 다만 XGBoost는 단순 Gradient Boosting와 달리 학습을 위한 목적식에 Regularization Term을 추가해 모델이 overfitting되는 것을 방지한다.
  * Regularization Term을 통해 XGBoost는 복잡한 모델에 패널티를 부여하여 overfitting을 방지한다.
* LightGBM
  * XGBoost와는 다르게 leaf-wise loss를 사용한다. (loss를 더 줄일 수 있음)
  * XGBoost 대비 2배 이상 빠른 속도, GPU 지원.
  * 대신 overfitting에 민감하여, 대량의 학습 데이터를 필요로 한다.
* CatBoost
  * 잔사 추정의 분산을 최소로 하면서 bias를 피하는 boosting 기법.
  * 관측치를 포함한 채로 boosting 하지 않고, 관측치를 뺀채로 학습해서 그 관측치에 대한 unbiased residual을 구하고 학습하는 형태.
  * Categorical Feature가 많은 경우 성능이 좋다고 알려져 있다.
  * Categorical Feature를 one-hot encoding이 아닌 수치형으로 변환한다.

-------------

### 실험 환경 & 소요 시간

* 실험 환경 : kaggle python nootbook (GPU)
* 소요 시간 : 약 3시간

-----------

### 코드

['./Tabular Playground Series - Sep 2021.py'](https://github.com/essential2189/ML_study/blob/main/kaggle/Tabular%20Playground%20Series%20-%20Sep%202021/Tabular%20Playground%20Series%20-%20Aug%202021.py)

-----------

### 참고자료

[XGBoost](https://xgboost.readthedocs.io/en/latest/)

[LightBGM](https://lightgbm.readthedocs.io/en/latest/)

[CatBoost](https://catboost.ai/)

