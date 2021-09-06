## Shopee - Price Match Guarantee

------------

### 결과

----------------

### 요약정보

* 도전기관 : 시큐레이어
* 도전자 : 왕승재
* 최종스코어 : 0.743
* 제출일자 : 2021-09-06
* 총 참여 팀 수 : 2426
* 순위 및 비율 : 85(3.5%)

### 결과화면

![결과](/Users/sengjeawang/Desktop/ML_study/kaggle/Shopee/screanshot/result.png)

![결과](/Users/sengjeawang/Desktop/ML_study/kaggle/Shopee/screanshot/score.png)

----------

### 사용한 방법 & 알고리즘

* NFnets-I0(from [timm](https://github.com/rwightman/pytorch-image-models)) + EfficientNet B5 essemble Model.
  * Adaptive Gradient Clipping(AGC).
  * 배치 정규화가 제공했던 정규화 효과를 대체하기 위해 드롭아웃을 사용.
  * [Sharpness-Aware Minimization(SAM)](https://arxiv.org/abs/2010.01412) 사용.
* ArcMarginProduct Module.
* 'eca_nfnet_l0'에 SiLU() 활성화가 포함되어 있으므로 Mish() 활성화로 대체했습니다. Mish() 활성화를 변경하는 이유는 여기서는 Ranger(RADAM + Lookahead) 최적화 프로그램을 사용하고 있으며 Mish() + Ranger 최적화 프로그램이 좋은 결과를 제공하기 때문입니다. (몇 가지 실험을 통해 틀릴 수 있음)

-------------

### 코드

./Shopee - Price Match Guarantee.py

-----------

### 참고자료

[NFnets](https://arxiv.org/abs/2102.06171)

[EfficientNet](https://arxiv.org/abs/1905.11946)



