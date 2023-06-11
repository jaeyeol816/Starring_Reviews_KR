## BERT 언어모델을 사용한 한국어 지도 리뷰 평점 예측 (Starring Reviews_KR)

사라진 네이버 지도 별점을 찾아서..


### 1. Introduction

[SKTBrain의 KoBERT](https://github.com/SKTBrain/KoBERT)모델을 확장하여 한국어로 된 장소(식당, 카페, 호텔 등)의 리뷰 데이터에 감정 분석(sentiment analysis) 프로젝트입니다. 다음 두 가지를 포함합니다.
- 최적의 성능을 갖도록 모델 설계
- Google Maps Platform API / Naver Search API를 활용한 실제 사용가능한 클라이언트-서버 구조 어플리케이션 개발

### 2. Model
#### 2-1. Dataset
- `Train/movie_train.txt` : 한국어 영화 리뷰 데이터셋. 0(부정)과 1(긍정)으로 라벨링됨 (from KoBERT, 약 20만개)
- `Train/place_train.txt`, `Train/place_test.txt` : 한국어 장소 리뷰 데이터셋. (약 2만개)

#### 2-2. Model Architecture
- BERT모델에 긍정/부정 Regression을 위한 2개의 hidden layer추가 (Linear -> ReLU -> Linear -> Sigmoid)
- MSE(Mean Squared Error) Loss, MAE (Mean Absolute Error) Inverse Accuracy

#### 2-3. Implementation
- pytorch 및 KoBERT 라이브러리 기반 구현
- 모델 클래스: `Train/train_1B.py`의 `BERTRegressor` 클래스
- Dataset 클래스: 
- Optimizor:
- Batch Size: 

### 3. System Architecture

### 4. Usage

### 5. Performance Analysis








