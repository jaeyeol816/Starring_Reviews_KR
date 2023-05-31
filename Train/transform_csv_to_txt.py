import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# CSV 파일 읽기
df = pd.read_csv('place_reviews_seperated.csv')

# '장소' column 삭제
df = df.drop(columns=['장소'])

# '리뷰'와 '별점' column의 이름 변경
df = df.rename(columns={'리뷰': 'document', '별점': 'label'})

# 'document'와 'label' column의 순서 변경
df = df[['document', 'label']]

# 겹치지 않는 랜덤 숫자 생성하여 'id' column 생성
df['id'] = np.random.choice(range(1, df.shape[0] + 1), size=df.shape[0], replace=False)

# 'id', 'document', 'label' column의 순서 변경
df = df[['id', 'document', 'label']]

# 데이터를 train set과 test set으로 나눔 (80%는 train set, 20%는 test set)
df_train, df_test = train_test_split(df, test_size=0.05, random_state=42)

# Train set과 test set을 txt 파일로 저장
df_train.to_csv('place_train.txt', sep='\t', index=False)
df_test.to_csv('place_test.txt', sep='\t', index=False)
