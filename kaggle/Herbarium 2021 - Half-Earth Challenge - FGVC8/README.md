## Herbarium 2021 - Half-Earth Challenge - FGVC8

------------

### 결과

----------------

### 요약정보

* 도전기관 : 시큐레이어
* 도전자 : 왕승재
* 최종스코어 : 0.0000
* 제출일자 : 2021-10-26
* 총 참여 팀 수 : 80
* 순위 및 비율 : 58 (72%)

### 결과화면

![결과](screanshot/score.PNG)

![결과](screanshot/leaderboard.PNG)

----------

### 사용한 방법 & 알고리즘

* ResNet34

  * ```python
    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    ResNet                                   --
    ├─Conv2d: 1-1                            9,408
    ├─BatchNorm2d: 1-2                       128
    ├─ReLU: 1-3                              --
    ├─MaxPool2d: 1-4                         --
    ├─Sequential: 1-5                        --
    │    └─BasicBlock: 2-1                   --
    │    │    └─Conv2d: 3-1                  36,864
    │    │    └─BatchNorm2d: 3-2             128
    │    │    └─ReLU: 3-3                    --
    │    │    └─Conv2d: 3-4                  36,864
    │    │    └─BatchNorm2d: 3-5             128
    │    └─BasicBlock: 2-2                   --
    │    │    └─Conv2d: 3-6                  36,864
    │    │    └─BatchNorm2d: 3-7             128
    │    │    └─ReLU: 3-8                    --
    │    │    └─Conv2d: 3-9                  36,864
    │    │    └─BatchNorm2d: 3-10            128
    │    └─BasicBlock: 2-3                   --
    │    │    └─Conv2d: 3-11                 36,864
    │    │    └─BatchNorm2d: 3-12            128
    │    │    └─ReLU: 3-13                   --
    │    │    └─Conv2d: 3-14                 36,864
    │    │    └─BatchNorm2d: 3-15            128
    ├─Sequential: 1-6                        --
    │    └─BasicBlock: 2-4                   --
    │    │    └─Conv2d: 3-16                 73,728
    │    │    └─BatchNorm2d: 3-17            256
    │    │    └─ReLU: 3-18                   --
    │    │    └─Conv2d: 3-19                 147,456
    │    │    └─BatchNorm2d: 3-20            256
    │    │    └─Sequential: 3-21             8,448
    │    └─BasicBlock: 2-5                   --
    │    │    └─Conv2d: 3-22                 147,456
    │    │    └─BatchNorm2d: 3-23            256
    │    │    └─ReLU: 3-24                   --
    │    │    └─Conv2d: 3-25                 147,456
    │    │    └─BatchNorm2d: 3-26            256
    │    └─BasicBlock: 2-6                   --
    │    │    └─Conv2d: 3-27                 147,456
    │    │    └─BatchNorm2d: 3-28            256
    │    │    └─ReLU: 3-29                   --
    │    │    └─Conv2d: 3-30                 147,456
    │    │    └─BatchNorm2d: 3-31            256
    │    └─BasicBlock: 2-7                   --
    │    │    └─Conv2d: 3-32                 147,456
    │    │    └─BatchNorm2d: 3-33            256
    │    │    └─ReLU: 3-34                   --
    │    │    └─Conv2d: 3-35                 147,456
    │    │    └─BatchNorm2d: 3-36            256
    ├─Sequential: 1-7                        --
    │    └─BasicBlock: 2-8                   --
    │    │    └─Conv2d: 3-37                 294,912
    │    │    └─BatchNorm2d: 3-38            512
    │    │    └─ReLU: 3-39                   --
    │    │    └─Conv2d: 3-40                 589,824
    │    │    └─BatchNorm2d: 3-41            512
    │    │    └─Sequential: 3-42             33,280
    │    └─BasicBlock: 2-9                   --
    │    │    └─Conv2d: 3-43                 589,824
    │    │    └─BatchNorm2d: 3-44            512
    │    │    └─ReLU: 3-45                   --
    │    │    └─Conv2d: 3-46                 589,824
    │    │    └─BatchNorm2d: 3-47            512
    │    └─BasicBlock: 2-10                  --
    │    │    └─Conv2d: 3-48                 589,824
    │    │    └─BatchNorm2d: 3-49            512
    │    │    └─ReLU: 3-50                   --
    │    │    └─Conv2d: 3-51                 589,824
    │    │    └─BatchNorm2d: 3-52            512
    │    └─BasicBlock: 2-11                  --
    │    │    └─Conv2d: 3-53                 589,824
    │    │    └─BatchNorm2d: 3-54            512
    │    │    └─ReLU: 3-55                   --
    │    │    └─Conv2d: 3-56                 589,824
    │    │    └─BatchNorm2d: 3-57            512
    │    └─BasicBlock: 2-12                  --
    │    │    └─Conv2d: 3-58                 589,824
    │    │    └─BatchNorm2d: 3-59            512
    │    │    └─ReLU: 3-60                   --
    │    │    └─Conv2d: 3-61                 589,824
    │    │    └─BatchNorm2d: 3-62            512
    │    └─BasicBlock: 2-13                  --
    │    │    └─Conv2d: 3-63                 589,824
    │    │    └─BatchNorm2d: 3-64            512
    │    │    └─ReLU: 3-65                   --
    │    │    └─Conv2d: 3-66                 589,824
    │    │    └─BatchNorm2d: 3-67            512
    ├─Sequential: 1-8                        --
    │    └─BasicBlock: 2-14                  --
    │    │    └─Conv2d: 3-68                 1,179,648
    │    │    └─BatchNorm2d: 3-69            1,024
    │    │    └─ReLU: 3-70                   --
    │    │    └─Conv2d: 3-71                 2,359,296
    │    │    └─BatchNorm2d: 3-72            1,024
    │    │    └─Sequential: 3-73             132,096
    │    └─BasicBlock: 2-15                  --
    │    │    └─Conv2d: 3-74                 2,359,296
    │    │    └─BatchNorm2d: 3-75            1,024
    │    │    └─ReLU: 3-76                   --
    │    │    └─Conv2d: 3-77                 2,359,296
    │    │    └─BatchNorm2d: 3-78            1,024
    │    └─BasicBlock: 2-16                  --
    │    │    └─Conv2d: 3-79                 2,359,296
    │    │    └─BatchNorm2d: 3-80            1,024
    │    │    └─ReLU: 3-81                   --
    │    │    └─Conv2d: 3-82                 2,359,296
    │    │    └─BatchNorm2d: 3-83            1,024
    ├─AdaptiveAvgPool2d: 1-9                 --
    ├─Linear: 1-10                           33,088,500
    =================================================================
    Total params: 54,373,172
    Trainable params: 54,373,172
    Non-trainable params: 0
    =================================================================
    ```

  ![model](screanshot/model.png)

-------------

### 실험 환경 & 소요 시간

* 실험 환경 : kaggle python nootbook (GPU)
* 소요 시간 : 약 2시간

-----------

### 코드

['./Herbarium 2021 - Half-Earth Challenge - FGVC8.py'](https://github.com/essential2189/AI_Competitions_2/blob/main/kaggle/Herbarium%202021%20-%20Half-Earth%20Challenge%20-%20FGVC8/Herbarium%202021%20-%20Half-Earth%20Challenge%20-%20FGVC8.py)

-----------

### 참고자료

[ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)

