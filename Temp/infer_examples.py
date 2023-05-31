
# 패키지 설치
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import ConcatDataset


# For saving
model_save_path = './models/regressor.pt'


# kobert 라이브러리 임포트
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model


# GPU 사용 세팅
# for CUDA
device = torch.device('cuda:0')
# for Mac
# device = torch.device('mps:0')

# bert 모델 불러오기   
bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")


# Tokenizer 가져오기
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


class BERTPredictDataset(Dataset):
	def __init__(self, sentence, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
		transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
		self.sentence = [transform(sentence)]	
		self.labels = [np.int32(0)]		# 무의미
	def __getitem__(self, index):
		return self.sentence[0]
	def __len__(self):
		return len(self.labels)	

max_len = 64

# 모델 클래스 정의
class BERTRegressor(nn.Module):
	def __init__(self,
							bert,
							hidden_size = 768,
							num_classes=2,
							dr_rate=None,
							params=None):
		super(BERTRegressor, self).__init__()
		self.bert = bert
		self.dr_rate = dr_rate

		self.lin1 = nn.Linear(hidden_size, 128)
		self.relu = nn.ReLU()
		self.lin2 = nn.Linear(128 , 1)
		self.sigmoid = nn.Sigmoid()
		if dr_rate:
				self.dropout = nn.Dropout(p=dr_rate)
	
	def gen_attention_mask(self, token_ids, valid_length):
		attention_mask = torch.zeros_like(token_ids)
		for i, v in enumerate(valid_length):
				attention_mask[i][:v] = 1
		return attention_mask.float()

	def forward(self, token_ids, valid_length, segment_ids):
		attention_mask = self.gen_attention_mask(token_ids, valid_length)
		
		_, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
		if self.dr_rate:
				out = self.dropout(pooler)
		else:
				out = pooler
		out = self.lin1(out)
		out = self.relu(out)
		out = self.lin2(out)
		out = self.sigmoid(out)
		return out


# 모델 로드하기
model = BERTRegressor(bertmodel, dr_rate=0.5).to(device)
model.load_state_dict(torch.load(model_save_path, map_location='cuda:0'))

# 새로운 문장에 대해 쿼리하기
def query_rating(sentence):
    sample_dataset = BERTPredictDataset(sentence, 0, 1, tok, max_len, True, False)
    sample_dataloader =  torch.utils.data.DataLoader(sample_dataset, batch_size=1, num_workers=5)
    it = iter(sample_dataloader)
    token_ids, valid_length, segment_ids = next(it)
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    model.eval()
    with torch.no_grad():
        out = model(token_ids, valid_length, segment_ids)
    scaled_out = out[0].item() * 4 + 1  # Scale the output to be between 1 and 5
    rounded_scaled_out = round(scaled_out, 2)  # Round the scaled output to 2 decimal places
    return rounded_scaled_out

print(query_rating('정말 최고의 식당입니다. 완전 강추!!😝'))
print(query_rating('괜찮긴 한데 가격이 좀 비싸요ㅠㅠ😐'))
print(query_rating('다시는 안 갈 것 같아😡'))
print(query_rating('꽤 괜찮았어요. 자주 가고 싶습니다'))
print(query_rating('가게 분위기는 좋고 청결했지만 좀 짰다..'))


