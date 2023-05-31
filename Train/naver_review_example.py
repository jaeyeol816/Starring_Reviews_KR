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



# kobert 라이브러리 임포트
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model


# transformers 라이브러리 임포트
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

# GPU 사용 세팅
device = torch.device('cuda:0')

# 모델 불러오기   
bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")

# 데이터셋 다운로드
os.system('wget -O .cache/ratings_train.txt http://skt-lsl-nlp-model.s3.amazonaws.com/KoBERT/datasets/nsmc/ratings_train.txt')
os.system('wget -O .cache/ratings_test.txt http://skt-lsl-nlp-model.s3.amazonaws.com/KoBERT/datasets/nsmc/ratings_test.txt')

# 데이터셋 불러오기
dataset_train = nlp.data.TSVDataset(".cache/ratings_train.txt", field_indices=[1,2], num_discard_samples=1)
dataset_test = nlp.data.TSVDataset(".cache/ratings_test.txt", field_indices=[1,2], num_discard_samples=1)

# Tokenizer 가져오기
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# 데이터셋 클래스 정의
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

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
batch_size = 64
warmup_ratio = 0.1
num_epochs = 1
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

# 데이터셋 인스턴스화
data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)


# DataLoader 인스턴스화
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)


# 모델 클래스 정의
class BERTClassifier(nn.Module):
    def __init__(self,
                bert,
                hidden_size = 768,
                num_classes=2,
                dr_rate=None,
                params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
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
        return self.classifier(out)


# 모델 인스턴스화
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)


# optimizer와 scheduler를 위한 세팅
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# optimizer, loss 인스턴스화
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# iteration 관련 세팅
t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

# Scheduler 인스턴스화
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


# accuracy 계산 함수
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


# Training Loop
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))


# 새로운 문장에 대해 쿼리하기
def infer(sentence):
    sample_dataset = BERTPredictDataset(sentence, 0, 1, tok, max_len, True, False)
    sample_dataloader =  torch.utils.data.DataLoader(sample_dataset, batch_size=1, num_workers=5)
    it = iter(sample_dataloader)
    token_ids, valid_length, segment_ids= next(it)
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    model.eval()
    with torch.no_grad():
        out = model(token_ids, valid_length, segment_ids)
    print(out)
    return out[0,1].item()

print(infer('정말 최고의 식당입니다. 완전 강추!!😝'))
print(infer('괜찮긴 한데 가격이 좀 비싸요ㅠㅠ😐'))
print(infer('다시는 안 갈 것 같아😡'))



