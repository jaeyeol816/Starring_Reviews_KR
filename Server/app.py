from flask import Flask, jsonify, request
from flask_cors import CORS

from infer import infer_list
from get_naver_blog_reviews import get_naver_blog_reviews
from get_google_map_reviews	import get_google_map_reviews

app = Flask(__name__)	
CORS(app)


@app.post('/predict')
def predict():
	data = request.json
	try:
		keyword = data['keyword']
		print(keyword)
	except Exception as e:
		print(str(e))
		return jsonify({'error': 'error code 1- 검색어를 입력하세요'})
	try:
		naver_list = get_naver_blog_reviews(keyword)
		google_list = get_google_map_reviews(keyword)
		review_list = naver_list + google_list
		# print(review_list)
	except Exception as e:
		print(str(e))
		return jsonify({'error': 'error code 2- 리뷰가 검색되지 않았습니다. 다른 검색어를 입력하세요'})
	try:
		result = infer_list(review_list)
		new_outputs = {t: s for t, s in result['outputs']}
		new_result = {'avg': float(result['avg']), 'outputs': new_outputs}
		print(new_result)		# test code
		return jsonify(new_result)
	except Exception as e:
		print(str(e))
		return jsonify({'error': 'error code 3- 유효하지 않음. 다른 검색어를 입력하세요'})

@app.route('/')
def route():
	return jsonify({'a': 'b'})

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=80, debug=True)