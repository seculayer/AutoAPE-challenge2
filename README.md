# AI Competitions 2

Competitions 2차년도(2021)

## Contribute

1. [kaggle](#kaggle-구조)/dacon 안에 구조를 맞추어서, 참가한 대회 내용을 작성합니다.
2. Pull Request를 작성합니다.

## Kaggle 구조

Kaggle 폴더 구조입니다.

```console
kaggle/
├── titanic/ # 예시 대회
│  ├── metadata.yaml
│  ├── README.md
│  └── *
└── {kaggle-competition-id}/
   ├── metadata.yaml
   ├── README.md
   └── *
```

1. Kaggle 대회 ID로 폴더를 생성합니다.

    Kaggle 대회 ID는 URL에서 찾을 수 있습니다.

    예시) 대회 URL이 `https://www.kaggle.com/c/acea-water-prediction`이면, `acea-water-prediction` 부분이 대회 ID입니다.

2. `metadata.yaml`과 `README.md`를 작성합니다.

    metadata에는 아래와 같은 내용들이 있어야한다.

    ```yaml
    id: kaggle-competition-id
    score:
    team:
    rank:
    date:
    organization:
    author:
      id:
      name:
    ```

    README에는 다음과 같은 내용이 있어야 합니다.

    - 결과 요약
    - 리더보드 이미지
    - 알고리즘, 문제 해결 방법
    - 참고 자료
    - 기타

3. 사용했던 코드들을 생성했던 폴더에 넣습니다.

## 참고

- [YAML](https://yaml.org/)
  - [YAML - 위키백과](https://ko.wikipedia.org/wiki/YAML)
- [Pro Git: 6.2 GitHub - GitHub 프로젝트에 기여하기](https://git-scm.com/book/ko/v2/GitHub-GitHub-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%EC%97%90-%EA%B8%B0%EC%97%AC%ED%95%98%EA%B8%B0)
