import requests
import json
import re
from dotenv import load_dotenv
import os

def get_naver_blog_reviews(query):
	try:
		load_dotenv()

		url = 'https://openapi.naver.com/v1/search/blog.json'
		display = 10
		start = 1
		sort = 'sim'
		client_id = os.getenv('NAVER_CLIENT_ID')
		client_secret = os.getenv('NAVER_CLIENT_SECRET')

		headers = {
				'X-Naver-Client-Id': client_id,
				'X-Naver-Client-Secret': client_secret
		}

		params = {
				'query': query,
				'display': display,
				'start': start,
				'sort': sort
		}

		response = requests.get(url, headers=headers, params=params)
		data = response.json()

		# description 추출 및 리스트에 저장
		descriptions = [re.sub('<b>|</b>', '', item['description']) for item in data['items']]
	except:
		descriptions = []
	return descriptions