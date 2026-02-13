import pandas as pd

df = pd.read_csv("processed_data/train_combined_sep.csv")
df_test = pd.read_csv("processed_data/test_cleaned_sep.csv")


from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# 1. 레이블 인코더 생성 및 학습
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])
# 클래스-숫자 매핑 결과 저장 (나중에 추론 시 결과 해석을 위해 필요)
mapping = dict(zip(range(len(le.classes_)), le.classes_))
print("📌 레이블 매핑 결과:", mapping)

# 2. 파이토치 텐서로 변환
# 앞서 만든 input_ids와 attention_masks를 함께 사용합니다.
train_labels = torch.tensor(df['label'].values)
print(f"레이블 텐서 생성 완료: {train_labels.shape}")

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

# 1. 환경 설정
CHECKPOINT_NAME = "klue/bert-base"
tokenizer_pretrained = CHECKPOINT_NAME

# 2. 데이터 분할 (Stratify 적용)
# 학습 데이터와 검증 데이터를 8:2 비율로 나눕니다.
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']  # 클래스 비율 유지
)

# 3. 커스텀 데이터셋 클래스 (제공해주신 코드 수정)
class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained):
        self.data = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 컬럼명에 맞춰 수정: 'document' -> 'conversation'
        sentence = self.data.iloc[idx]['conversation']
        label = self.data.iloc[idx]['label']

        tokens = self.tokenizer(
            str(sentence),           # 문장 전달
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=512,          # 길이를 명시적으로 지정
            add_special_tokens=True
        )

        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        token_type_ids = torch.zeros_like(attention_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }, torch.tensor(label, dtype=torch.long)

# 4. Dataset 인스턴스 생성
train_dataset = TokenDataset(train_df, tokenizer_pretrained)
val_dataset = TokenDataset(val_df, tokenizer_pretrained)

# 5. DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)

print(f"✅ DataLoader 구축 완료 (Stratify 적용)")
print(f"학습 데이터 개수: {len(train_df)} | 검증 데이터 개수: {len(val_df)}")

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 디바이스: {device}')

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertModel


class CustomBertModel(nn.Module):
    def __init__(self, bert_pretrained, dropout_rate=0.5):
        super(CustomBertModel, self).__init__()
        # 사전학습 모델 지정
        self.bert = BertModel.from_pretrained(bert_pretrained)
        self.dr = nn.Dropout(p=dropout_rate)
        # 5 class 분류
        self.fc = nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = output['last_hidden_state']
        # last_hidden_state[:, 0, :]는 [CLS] 토큰을 가져옴
        x = self.dr(last_hidden_state[:, 0, :])
        # FC 을 거쳐 최종 출력
        x = self.fc(x)
        return x

bert = CustomBertModel(CHECKPOINT_NAME)
bert.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(bert.parameters(), lr=1e-5)

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

# 클래스 이름 정의 (파일 헤더 및 출력용)
target_names = ['협박', '갈취', '직장 내 괴롭힘', '기타 괴롭힘', '일반 대화']

def model_train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0
    corr = 0
    counts = 0

    all_preds = []
    all_labels = []

    progress_bar = tqdm(data_loader, unit='batch', total=len(data_loader), mininterval=1)

    for idx, (inputs, labels) in enumerate(progress_bar):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(**inputs)
        logits = output.logits if hasattr(output, 'logits') else output

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        _, pred = logits.max(dim=1)

        corr += pred.eq(labels).sum().item()
        counts += len(labels)
        running_loss += loss.item() * labels.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 진행 바 업데이트
        current_f1 = f1_score(all_labels, all_preds, average='weighted')
        progress_bar.set_description(f"Loss: {running_loss/counts:.4f}, Acc: {corr/counts:.4f}, F1: {current_f1:.4f}")

    final_loss = running_loss / len(data_loader.dataset)
    final_acc = corr / len(data_loader.dataset)
    # 🔹 클래스별 F1 Score 계산 (average=None)
    final_f1_per_class = f1_score(all_labels, all_preds, average=None)
    final_f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    return final_loss, final_acc, final_f1_weighted, final_f1_per_class

def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0
    corr = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            output = model(**inputs)
            logits = output.logits if hasattr(output, 'logits') else output

            _, pred = logits.max(dim=1)
            corr += torch.sum(pred.eq(labels)).item()
            running_loss += loss_fn(logits, labels).item() * labels.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_loss = running_loss / len(data_loader.dataset)
    final_acc = corr / len(data_loader.dataset)
    # 🔹 클래스별 F1 Score 계산
    final_f1_per_class = f1_score(all_labels, all_preds, average=None)
    final_f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    return final_loss, final_acc, final_f1_weighted, final_f1_per_class

# --- 실행부 ---
num_epochs = 20
model_name = 'bert-kor-base'
min_loss = np.inf
metric_path = f"{model_name}_metrics.txt"

# 🔹 파일 헤더 수정 (클래스별 F1 컬럼 추가)
f1_headers = "\t".join([f"T_F1_{name}" for name in target_names] + [f"V_F1_{name}" for name in target_names])
with open(metric_path, 'w') as f:
    f.write(f"Epoch\tTrain_Loss\tTrain_Acc\tTrain_F1_W\tVal_Loss\tVal_Acc\tVal_F1_W\t{f1_headers}\n")

for epoch in range(num_epochs):
    # Training
    train_loss, train_acc, train_f1_w, train_f1_class = model_train(
        bert, train_loader, loss_fn, optimizer, device
    )

    # Evaluation
    val_loss, val_acc, val_f1_w, val_f1_class = model_evaluate(
        bert, val_loader, loss_fn, device
    )

    # Checkpoint (Validation Loss 기준)
    if val_loss < min_loss:
        print(f'✨ [INFO] val_loss improved to {val_loss:.5f}. Saving Model!')
        min_loss = val_loss
        torch.save(bert.state_dict(), f'/home/summer24/DataFrom101/ddd/DKTC_classification/태경/result/{model_name}.pth')

    # 콘솔 출력 (전체 요약)
    print(f'Epoch [{epoch+1:02d}/{num_epochs}]')
    print(f'TRAIN | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1(W): {train_f1_w:.4f}')
    print(f'VALID | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1(W): {val_f1_w:.4f}')
    
    # 클래스별 F1 점수 상세 출력
    print("Class F1 (Val): " + " | ".join([f"{name}: {score:.4f}" for name, score in zip(target_names, val_f1_class)]))
    print('-' * 50)

    # 🔹 TXT 파일에 저장
    train_f1_str = "\t".join([f"{s:.4f}" for s in train_f1_class])
    val_f1_str = "\t".join([f"{s:.4f}" for s in val_f1_class])
    
    with open(metric_path, 'a') as f:
        f.write(
            f"{epoch+1}\t"
            f"{train_loss:.4f}\t{train_acc:.4f}\t{train_f1_w:.4f}\t"
            f"{val_loss:.4f}\t{val_acc:.4f}\t{val_f1_w:.4f}\t"
            f"{train_f1_str}\t{val_f1_str}\n"
        )