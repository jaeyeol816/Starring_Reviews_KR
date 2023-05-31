import os
from kobert.utils.aws_s3_downloader import AwsS3Downloader

def download(path, s3_urls):
  path_full = os.path.join(os.getcwd(), path)
  os.makedirs(path, exist_ok=True)
  
  s3 = AwsS3Downloader()
  for key, url in s3_urls.items():
    file_path = os.path.join(path_full)
    s3.download(url, file_path)
  


urls = {'config': 's3://starring-reviews-bucket/config.json',
        'vocab': 's3://starring-reviews-bucket/kobert_news_wiki_ko_cased-1087f8699e.spiece',
        'bert': 's3://starring-reviews-bucket/pytorch_model.bin',
        'regressor': 's3://starring-reviews-bucket/regressor.pt'}
download('models', urls)