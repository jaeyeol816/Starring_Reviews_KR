## BERT 언어모델을 사용한 한국어 지도 리뷰 평점 예측 (Starring Reviews_KR)

사라진 네이버 지도 별점을 찾아서..🙄


### 1. Introduction

[SKTBrain의 KoBERT](https://github.com/SKTBrain/KoBERT)모델을 확장하여 한국어로 된 장소(식당, 카페, 호텔 등)의 리뷰 데이터에 감정 분석(sentiment analysis) 프로젝트입니다. 다음 두 가지를 포함합니다.
- 최적의 성능을 갖도록 모델을 설계하는 실험 과정
- Google Maps Platform API / Naver Search API를 활용한 실제 사용가능한 클라이언트-서버 구조 어플리케이션 개발


### 2. Model
#### 2-1. Dataset
- `Train/movie_train.txt` : 한국어 영화 리뷰 데이터셋. 0(부정)과 1(긍정)으로 라벨링됨 (from KoBERT, 약 20만개)
- `Train/place_train.txt`, `Train/place_test.txt` : 한국어 장소 리뷰 데이터셋. (약 2만개)

#### 2-2. Model Architecture
![Screenshot 2023-06-13 at 11 22 02 PM](https://github.com/jaeyeol816/Starring_Reviews_KR/assets/80497842/9449d492-d5e4-44df-ae39-51461326e4f1)
- BERT모델에 긍정/부정 Regression을 위한 2개의 hidden layer추가 (Linear -> ReLU -> Linear -> Sigmoid)
- MSE(Mean Squared Error) Loss, MAE (Mean Absolute Error) Inverse Accuracy

#### 2-3. Implementation
- pytorch 및 KoBERT 라이브러리 기반 구현
- 모델 클래스: `BERTRegressor` 클래스
- Training 코드: `Train/train1A.py`, `Train/train1B.py`, `Train/train2C.py`, `Train/train2D.py` (각 조건에 대한 설명은 5절 참고)

### 3. System Architecture
![Screenshot 2023-06-13 at 11 35 33 PM](https://github.com/jaeyeol816/Starring_Reviews_KR/assets/80497842/581428fb-9907-43b5-8297-e5accca7fe1b)

### 4. Usage
#### 4-1. Training 
- CUDA 11.X 버전이 설치된 환경에서 실행 권장
- (1) 리포지토리를 clone한 후 Train디렉토리로 이동합니다.
  - `git clone https://github.com/jaeyeol816/Starring_Reviews_KR.git`
  - `cd Train`
- (2) Anaconda 가상환경을 만듭니다. (파이썬 3.9버전 권장)
  - 예: `conda create --name starring_train python=3.9`
- (3) 필요한 패키지를 설치합니다.
  - `pip install -r requirements.txt`
- (4) training 코드를 실행합니다.
  - 예: `python train_1B.py`
> 데이터셋을 직접 수집하는 방법
>   - Google Maps API를 사용하기 때문에 API키가 필요합니다. 키를 발급받은 후 아래 내용과 같이 `Train/.env` 파일을 만듭니다.
>   ```
>   GOOGLE_API_KEY={발급받은 키}
>   ```
>   - `Train/keywords_for_query.txt`파일을 생성한 후, 리뷰를 수집할 식당/카페/호텔 등의 이름들을 한 줄에 하나씩 작성합니다.
>   - 이후 다음과 같은 코드를 실행시키면 `txt` 포맷의 데이터셋 (`place_train.txt`, `place_test.txt`)이 생성됩니다.
>   - `python create_or_append_dataset.py`
>   - `python seperate_by_sentence.py`
>   - `python transform_csv_to_txt.py`

#### 4-2. Running the Server
- CUDA 11.X 버전이 설치된 환경에서 실행 권장
- (1) 리포지토리
  - `git clone https://github.com/jaeyeol816/Starring_Reviews_KR.git`
  - `cd Server`
- (2) Anaconda 가상환경을 만듭니다. (파이썬 3.9버전 권장)
  - `conda create --name starring_server python=3.9`


### 5. Performance Analysis








