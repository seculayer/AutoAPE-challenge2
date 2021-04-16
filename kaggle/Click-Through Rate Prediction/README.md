# Click-Through Rate Prediction

## 결과

### 요약정보

- 도전기관 : 시큐레이어
- 도전자 : 김연진
- 최종스코어 : 0.8549
- 제출일자 : 2021-04-07
- 총 참여 팀 수 : 1602
- 순위 및 비율 : 1557(97.19%)

### 결과화면

![ctr_score](./img/ctr_score.PNG)
![ctr_rank](./img/ctr_rank.png)

## 사용한 방법 & 알고리즘
방법1. 
- user 암호화 정보 ( c id ) drop
- hour 변수 시간/날짜/요일 split
- 라벨 인코딩
- 노멀라이징
- csv 파일 분할 / 각 파일마다 10,000개씩 끊어서 전처리 / 멀티프로세싱
- DNN 모델 사용 

방법2. 
- 총 데이터 중 20% 만 랜덤 샘플링 사용
- 중복 데이터 제거 
- hour 변수 시간/날짜/요일 split
- banner_pos history 생성 ex ) 0 클릭 => 1 클릭 => 2 클릭  (012)
- 해시함수를 이용한 인코딩 ex ) aek39ga14f => 1029384092 => 384092 
- 적은 수의 카테고리 피쳐 ( banner_pos )  원핫 인코딩
- 노멀라이징
- FM + DNN 모델 사용 

## 코드

['./main.py'](./main.py)
['./data_anal.py'](./data_anal.py)
['./features.py'](./features.py)
['./preprocess.py'](./preprocess.py)
['./train.py'](./train.py)
['./my_models/app_model.py'](./my_models/app_model.py)
['./my_models/banner_model.py'](./my_models/banner_model.py)
['./my_models/cid_model.py'](./my_models/cid_model.py)
['./my_models/date_model.py'](./my_models/date_model.py)
['./my_models/device_model.py'](./my_models/device_model.py)
['./my_models/site_model.py'](./my_models/site_model.py)

## 참고 자료
