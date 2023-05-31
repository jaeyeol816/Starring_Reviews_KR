
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


# bert 모델 불러오기   
bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")


# Tokenizer 가져오기
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


class BERTPredictDataset(Dataset):
	def __init__(self, sentences, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
		transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
		self.len = len(sentences)
		self.sentences = [transform(s) for s in sentences]
		self.labels = [(np.int32(0)) for s in sentences]
	def __getitem__(self, i):
			return (self.sentences[i] + (self.labels[i], ))
	def __len__(self):
			return (self.len)

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
def infer_list(reviews):
    reviews_dataset = BERTPredictDataset(reviews, 0, 1, tok, max_len, True, False)
    sample_dataloader =  torch.utils.data.DataLoader(reviews_dataset, batch_size=len(reviews_dataset), num_workers=5)
    it = iter(sample_dataloader)
    token_ids, valid_length, segment_ids, labels = next(it)
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    model.eval()
    with torch.no_grad():
        outs = model(token_ids, valid_length, segment_ids)
    np_outs = outs.to('cpu').numpy()
    scaled_out = np_outs* 4 + 1  # Scale the output to be between 1 and 5
    rounded_scaled_outs = np.round(scaled_out, 2)  # Round the scaled output to 2 decimal places
    avg = np.round(np.average(rounded_scaled_outs), 2)
    outputs = [(s, round(float(o),2)) for s, o in zip(reviews, rounded_scaled_outs)]
		# 이제 딕셔너리를 생성합니다.
    result = {'avg': avg, 'outputs': outputs}
    return result
