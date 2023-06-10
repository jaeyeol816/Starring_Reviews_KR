import requests
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
import pandas as pd

dataset_file_name = 'place_reviews.csv'

if os.path.isfile(dataset_file_name):
	df = pd.read_csv(dataset_file_name)
else:
	df = pd.DataFrame(columns=['장소', '별점', '리뷰'])

fetch_keywords = ["Eats", "시오미스시", "카페이유", "행복한식탁", "개나리아구찜강동점", "고추장구이",
		  "서초면옥 천호점", "맥도날드 굽은다리역DT점", "다다감자탕", "이한진 숙성회", "마포숯불갈비",
			"라베이크두번째이야기", "시오미스시", "큰손칼국수", "다다감자탕", "서초면옥 천호점", "아구타운",
			]

with open('keywords_for_query.txt', 'r', encoding='utf-8') as f:
	fetch_keywords = []
	for line in f:
		fetch_keywords.append(line.strip())	

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

for i, keyword in enumerate(fetch_keywords):

	query = quote_plus(keyword)

	# Task1) find place
	url1 = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={query}&inputtype=textquery&key={api_key}"
	payload = {}
	headers = {}

	response = requests.request("GET", url1, headers=headers, data=payload)
	response_json = response.json()

	if len(response_json['candidates']) == 0:
		print(f'Cannot found place for {keyword} keyword')
		continue

	place_id = response_json['candidates'][0]['place_id']
	print(f'{keyword}의 place id: {place_id}')

	# Task2) find reviews and ratings
	url2 = f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews%2Crating&user_ratings_total=9&reviews_no_translations=true&language=ko&key={api_key}'
	payload={}
	headers = {}

	response = requests.request("GET", url2, headers=headers, data=payload)

	response_json = response.json()
	
	if 'result' in response_json and 'reviews' in response_json['result']:
		for review in response_json['result']['reviews']:
			# 'rating'과 'text' key가 각각 존재하는지 확인
			if 'rating' in review and 'text' in review and review['text']:
				# DataFrame에 추가
				new_row = pd.DataFrame([{'장소': keyword, '별점': review['rating'], '리뷰': review['text']}])
				df = pd.concat([df, new_row], ignore_index=True)
	else:
		print(f"Invalid data for {keyword} keyword.")
		continue

df.to_csv(dataset_file_name, index=False)
