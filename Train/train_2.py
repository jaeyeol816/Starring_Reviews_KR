### train_2.py
# 실험조건) 영화리뷰로 BERT 먼저 fine tuning하고, 이후에 장소 리뷰로 또 fine tuning
# output layer: lin + relu + lin + sigmoid
# loss: MSE
# accuracy: MAE 역수


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
from torch.utils.tensorboard import SummaryWriter


# For TensorBoard
writer = SummaryWriter('runs/train2')

# For saving
model_save_path = './models/train_2.pt'


# kobert 라이브러리 임포트
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model


# transformers 라이브러리 임포트
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

# GPU 사용 세팅
# for CUDA
device = torch.device('cuda:1')
# for Mac
# device = torch.device('mps:0')

# 모델 불러오기   
bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")


# 데이터셋 불러오기
movie_train = nlp.data.TSVDataset("./movie_train.txt", field_indices=[1,2], num_discard_samples=1)
place_train = nlp.data.TSVDataset("./place_train.txt", field_indices=[1,2], num_discard_samples=1)

movie_test = nlp.data.TSVDataset("./movie_test.txt", field_indices=[1,2], num_discard_samples=1)
place_test = nlp.data.TSVDataset("./place_test.txt", field_indices=[1,2], num_discard_samples=1)

# Tokenizer 가져오기
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# 데이터셋 클래스 정의
class BERTDataset(Dataset):
	def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair, type):
		transform = nlp.data.BERTSentenceTransform(
			bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
		self.sentences = [transform([i[sent_idx]]) for i in dataset]
		if (type == 'movie'):
			self.labels = [np.int32(i[label_idx]) for i in dataset]
		else:
			self.labels = [(np.int32(i[label_idx]) - 1) / 4 for i in dataset]
	def __getitem__(self, i):
			return (self.sentences[i] + (self.labels[i], ))
	def __len__(self):
			return (len(self.labels))

class BERTPredictDataset(Dataset):
	def __init__(self, sentence, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
		transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
		self.sentence = [transform(sentence)]	
		self.labels = [np.int32(0)]		# 무의미
	def __getitem__(self, index):
		return self.sentence[0]
	def __len__(self):
		return len(self.labels)	


# hyperparameter 세팅
max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 2
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

# 데이터셋 인스턴스화
data_movie_train = BERTDataset(movie_train, 0, 1, tok, max_len, True, False, 'movie')
data_place_train = BERTDataset(place_train, 0, 1, tok, max_len, True, False, 'place')
# data_movie_test = BERTDataset(movie_test, 0, 1, tok, max_len, True, False, 'movie')
data_test = BERTDataset(place_test, 0, 1, tok, max_len, True, False, 'place')
# data_test = ConcatDataset([data_movie_test, data_place_test])


# DataLoader 인스턴스화
train_movie_dataloader = torch.utils.data.DataLoader(data_movie_train, batch_size=batch_size, num_workers=5)
train_place_dataloader = torch.utils.data.DataLoader(data_place_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=2, num_workers=5)


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


# 모델 인스턴스화
model = BERTRegressor(bertmodel, dr_rate=0.5).to(device)

# For TensorBoard
# writer.add_graph(model)
# writer.close()

# optimizer와 scheduler를 위한 세팅
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


# optimizer, loss 인스턴스화
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.MSELoss()

# iteration 관련 세팅
t_total = (len(train_movie_dataloader)+len(train_place_dataloader)) * num_epochs
warmup_step = int(t_total * warmup_ratio)

# Scheduler 인스턴스화
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


# accuracy 계산 함수 (MAE의 역수)
def calc_accuracy(X,Y):
	mae = torch.mean(torch.abs(X.squeeze(-1) - Y.float())).item()
	epsilon = 1e-7  # Add a small constant for numerical stability
	return 1.0 / (mae + epsilon)


# Training Loop (With Movie Training Dataset)
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_movie_dataloader), total=len(train_movie_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out.squeeze(-1), label.float())  # Cast labels to float and adjust the dimensions
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("movie epoch {} batch id {} loss {} train acc(MAE inverse) {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
	    			# for TensorBoard
            writer.add_scalar('training loss', loss.data.cpu().numpy(), e+1 + batch_id+1)
            # writer.close()
            writer.add_scalar('training accuracy (MAE)', train_acc / (batch_id+1), e+1 + batch_id+1)
            # writer.close()
    print("movie epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("movie epoch {} test acc(MAE) {}".format(e+1, test_acc / (batch_id+1)))
    writer.add_scalar('test accuracy (MAE)', test_acc / (batch_id+1), e+1)
    # writer.close()

n_iter_after_moive = num_epochs * len(train_movie_dataloader)	

# Training Loop (with Place Trianing Dataset)
for e in range(num_epochs):
	train_acc = 0.0
	test_acc = 0.0
	model.train()
	for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_place_dataloader), total=len(train_place_dataloader)):
			optimizer.zero_grad()
			token_ids = token_ids.long().to(device)
			segment_ids = segment_ids.long().to(device)
			valid_length= valid_length
			label = label.long().to(device)
			out = model(token_ids, valid_length, segment_ids)
			loss = loss_fn(out.squeeze(-1), label.float())  # Cast labels to float and adjust the dimensions
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
			optimizer.step()
			scheduler.step()  # Update learning rate schedule
			train_acc += calc_accuracy(out, label)
			if batch_id % log_interval == 0:
					print("place epoch {} batch id {} loss {} train acc(MAE inverse) {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
					# for TensorBoard
					writer.add_scalar('training loss', loss.data.cpu().numpy(), n_iter_after_moive + e+1 + batch_id+1)
					# writer.close()
					writer.add_scalar('training accuracy (MAE)', train_acc / (batch_id+1), n_iter_after_moive + e+1 + batch_id+1)
					# writer.close()
	print("place epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
	model.eval()
	for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
			token_ids = token_ids.long().to(device)
			segment_ids = segment_ids.long().to(device)
			valid_length= valid_length
			label = label.long().to(device)
			out = model(token_ids, valid_length, segment_ids)
			test_acc += calc_accuracy(out, label)
	print("place epoch {} test acc(MAE) {}".format(e+1, test_acc / (batch_id+1)))
	writer.add_scalar('test accuracy (MAE)', test_acc / (batch_id+1), n_iter_after_moive + e+1)
	# writer.close()



# 모델 저장하기
torch.save(model.state_dict(), model_save_path)

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


