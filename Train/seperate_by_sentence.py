import pandas as pd
import re

# 파일 로드
df = pd.read_csv('place_reviews.csv')

# 새로운 데이터프레임 생성
new_df = pd.DataFrame(columns=df.columns)
last_place = None  # 마지막 장소 기억을 위한 변수

# 각 행에 대해
for idx, row in df.iterrows():
    # 장소 중복 방지
    if last_place == row['장소']:
        row['장소'] = ""
    else:
        last_place = row['장소']
    
    # 리뷰를 문장으로 분리하기 위한 정규표현식 패턴 정의
    pattern = r"\.\s|\?\s|\!\s|\n|\^\^\s"
    sentences = re.split(pattern, row['리뷰'])
    
    for sentence in sentences:
        # 이모티콘 분리
        emo_pattern = r"[\u2764|\u263A|\u2728|\uD83D\uDE0A|\uD83D\uDE03|\uD83D\uDE0D|\uD83D\uDE18|\uD83D\uDE1A|\uD83D\uDE17|\uD83D\uDE19|\uD83D\uDE1C|\uD83D\uDE1D|\uD83D\uDE1B|\uD83D\uDE33|\uD83D\uDE01|\uD83D\uDE14|\uD83D\uDE02|\uD83D\uDE13|\uD83D\uDE4D|\uD83D\uDE45]"
        emo_sentences = re.split(emo_pattern, sentence)
        emo_sentences = [emo_sentence.strip() for emo_sentence in emo_sentences if emo_sentence.strip()]
        
        for emo_sentence in emo_sentences:
            new_df = new_df.append({
                '장소': row['장소'],
                '별점': row['별점'],
                '리뷰': emo_sentence
            }, ignore_index=True)

# 새로운 csv 파일로 저장
new_df.to_csv('place_reviews_seperated.csv', index=False)

# '장소'가 연속해서 등장하는 경우 첫 항목만 남기고 나머지 삭제
# 파일 로드
df = pd.read_csv('place_reviews_seperated.csv')
# '장소' column에 대해 중복 삭제
df['장소'] = df['장소'].mask(df['장소'].duplicated(), '')
# 결과를 새로운 csv 파일로 저장
df.to_csv('place_reviews_seperated.csv', index=False)
