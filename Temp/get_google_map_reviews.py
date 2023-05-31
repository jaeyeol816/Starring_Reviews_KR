import requests
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
import json
import re

def get_google_map_reviews(query):
	load_dotenv()

	api_key = os.getenv('GOOGLE_API_KEY')
	query = quote_plus(query)

	# Task1) Find Place
	url1 = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={query}&inputtype=textquery&key={api_key}"

	payload={}
	headers = {}

	response = requests.request("GET", url1, headers=headers, data=payload)
	response_json = response.json()  # JSON 파싱


	# 예외처리 코드 추가
	if len(response_json['candidates']) == 0:
		pass

	place_id = response_json['candidates'][0]['place_id']  # place_id 추출

	# Task2) Find Reviews
	url2 = f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews&reviews_no_translations=true&language=ko&key={api_key}'
	payload={}
	headers = {}

	response = requests.request("GET", url2, headers=headers, data=payload)
	
	response_json = response.json()  # JSON 파싱
	
	# text 항목만 추출하고 "\n" 제거, 문장 단위로 분리
	review_texts = [sentence.strip() for review in response_json['result']['reviews'] for sentence in re.split(r'(?<=[.!?])\s+', review['text'].replace('\n', ' ')) if sentence]
	return review_texts

print(get_google_map_reviews('정돈 판교점'))
