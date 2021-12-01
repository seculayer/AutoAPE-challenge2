#Inclusive Images Challenge

## 결과

### 요약정보

- 도전기관 : 시큐레이어
- 도전자 : 이길상
- 최종스코어 : 0.08840
- 제출일자 : 2021-01-19
- 총 참여 팀 수 : 468
- 순위 및 비율 : 106(22.65%)

### 결과화면

![leaderboard](./img/leaderboard.png)

## 사용한 방법 & 알고리즘
1. 대회설명:
  
    이미지 학습에서 모델의 성능을 크게 좌우하는 데이터 세트를 개선하기 위해, 주어진 이미지 데이터들을 각각 지리적, 문화적 맥학으로 일반화 시키는 것을 통해 포괄적인 기계학습에서의 발전을 이루는 것이 목표이다.

2. 데이터 설명: 
 
    * stage_2_test_images.zip : 두번째 스테이지에 제공된 이미지 데이터셋이다. 99,649 개의 이미지로 이루어져 있으며, 각각의 이미지는 최대 1024픽셀을 넘지 못하게 크기가 조정되어 있다.
    
    * stage_1_test_images.zip : 첫번째 스테이지에 제공된 이미지 데이터셋이다. 32,580개의 이미지로 이루어져 있으며, 각각의 이미지는 최대 1024픽셀을 넘지 못하게 크기가 조정되어 있다.
    
    * train_bounding_boxes.csv : 훈련 이미지에서 감지된 맥락 영역을 나타내는 표 데이터이다. 이미지 id, 탐지된  LabelName,  영역범위를 나타내는 XMin, XMax, YMin, YMax 값, 감지 신뢰도를 나타내는 Confidence 값이 있다.
    
    * train_human_labels.csv : 훈련 이미지를 사람이 직접 보고 판별한 레이블 데이터이다. Confidence 값은 전부 1이다.
    
    * train_machine_labels.csv : 기계학습에 의해 판별된 훈련 이미지 레이블 데이터이다.
    
    * tuning_labels.csv : 테스트 이미지 세트 중1,000개 이미지에 대한 레이블 데이터이다.
     
    * class-descriptions.csv : 각 레이블 값이 의미하는 바를 서술한 표 데이터이다.

3. 알고리즘 설명:

     tuning_labels.csv의 데이터와 sample submission을 조합하여 제출하였다.
     
## 코드

['./src.py'](./src.py)

## 참고 자료

- 
