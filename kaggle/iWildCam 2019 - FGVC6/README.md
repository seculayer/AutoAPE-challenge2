# iWildCam 2019 - FGVC6

## 결과

### 요약정보

- 도전기관 : 시큐레이어
- 도전자 : 김연진
- 최종스코어 : 0.106
- 제출일자 : 2021-03-26
- 총 참여 팀 수 : 336
- 순위 및 비율 : 177(52.67%)

### 결과화면

![wildcam_score](./img/wildcam_score.PNG)
![wildcam_rank](./img/wildcam_rank.png)

## 사용한 방법 & 알고리즘
 
- user 암호화 정보 ( c id ) drop
- hour 변수 시간/날짜/요일 split
- 라벨 인코딩
- 노멀라이징
- csv 파일 분할 / 각 파일마다 10,000개씩 끊어서 전처리 / 멀티프로세싱
- DNN 모델 사용 

## 코드

['./main.py'](./main.py)
['./models.py'](./models.py)
['./preprocess.py'](./preprocess.py)
['./pca_detect.py'](./pca_detect.py)


## 참고 자료
