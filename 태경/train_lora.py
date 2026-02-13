import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report 
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# --- 0. 환경 설정 및 데이터 준비 ---
CHECKPOINT_NAME = "klue/bert-base"
BATCH_SIZE = 128
MAX_LEN = 512
EPOCHS = 500
LEARNING_RATE = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 수동 레이블 매핑 정의
class_to_id = {
    '협박 대화': 0,
    '갈취 대화': 1,
    '직장 내 괴롭힘 대화': 2,
    '기타 괴롭힘 대화': 3,
    '일반 대화': 4
}

# --- 1. 데이터셋 정의 ---
class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['conversation']
        label = self.data.iloc[idx]['label']
        
        tokens = self.tokenizer(
            str(text),
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            add_special_tokens=True
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'token_type_ids': torch.zeros(MAX_LEN, dtype=torch.long),
        }, torch.tensor(label, dtype=torch.long)

# --- 2. SupCon Loss (Supervised Contrastive Loss) 정의 ---
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (batch_size, feature_dim) -> 정규화된 상태여야 함
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(DEVICE)

        # 유사도 계산
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # 수치적 안정을 위해 최대값 차감
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 자기 자신을 제외한 마스크
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(DEVICE), 0)
        mask = mask * logits_mask

        # 지수 계산
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 양성 쌍에 대한 평균 로그 확률 계산
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss

# --- 3. LoRA 모델 구조 정의 ---
class CustomBertSupConModel(nn.Module):
    def __init__(self, bert_pretrained, dropout_rate=0.5):
        super(CustomBertSupConModel, self).__init__()
        # BERT 원본 로드
        self.bert = BertModel.from_pretrained(bert_pretrained)
        
        # 🔥 LoRA 설정
        lora_config = LoraConfig(
            r=32, 
            lora_alpha=64,
            target_modules=["query","value"], 
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.bert = get_peft_model(self.bert, lora_config)
        
        self.dr = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(768, 5) # 분류기
        self.projection_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 128) # SupCon용 프로젝션 헤드
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = output['last_hidden_state'][:, 0, :]
        
        logits = self.fc(self.dr(cls_token))
        proj_feat = F.normalize(self.projection_head(cls_token), dim=1) # 정규화된 벡터 반환
        return logits, proj_feat

def train():

    df = pd.read_csv("processed_data/train_combined_no_sep.csv")
    df['label'] = df['class'].map(class_to_id)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    os.makedirs("saved_datasets", exist_ok=True) # 폴더가 없으면 생성
    train_df.to_csv("saved_datasets/train_split.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv("saved_datasets/val_split.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 학습/검증 데이터셋이 'saved_datasets/' 폴더에 저장되었습니다.")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_NAME)
    train_loader = DataLoader(TokenDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TokenDataset(val_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False)

    model = CustomBertSupConModel(CHECKPOINT_NAME).to(DEVICE)
    model.bert.print_trainable_parameters()

    criterion_ce = nn.CrossEntropyLoss()
    criterion_supcon = SupConLoss(temperature=0.07)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_f1 = 0.0
    patience = 5
    counter = 0
    
    id_to_class = {v: k for k, v in class_to_id.items()}
    target_names = [id_to_class[i] for i in range(len(class_to_id))]

    # 🔥 Metric 파일 헤더 수정 (클래스별 F1 컬럼 추가)
    metric_file = "training_metrics_detailed.txt"
    class_f1_headers = "\t".join([f"Val_F1_{name}" for name in target_names])
    with open(metric_file, "w", encoding="utf-8-sig") as f:
        f.write(f"Epoch\tTrain_Loss\tTrain_Acc\tTrain_F1\tVal_Acc\tVal_F1\t{class_f1_headers}\n")

    for epoch in range(EPOCHS):
        # --- Train 단계 ---
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits, proj_feat = model(**inputs)
            
            loss = criterion_ce(logits, labels) + 0.1 * criterion_supcon(proj_feat, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 🔥 이 줄 추가
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Train 지표 계산
        avg_train_loss = train_loss / len(train_loader)
        train_acc = (np.array(train_preds) == np.array(train_labels)).mean() * 100
        train_f1 = f1_score(train_labels, train_preds, average='macro') * 100

        # --- Validation 단계 ---
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                logits, _ = model(**inputs)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Val 지표 계산
        val_acc = (np.array(val_preds) == np.array(val_labels)).mean() * 100
        val_f1_macro = f1_score(val_labels, val_preds, average='macro') * 100
        # 🔥 클래스별 F1-Score 추출
        val_f1_per_class = f1_score(val_labels, val_preds, average=None) * 100 

        # 콘솔 출력
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"TRAIN | Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}%")
        print(f"VAL   | Acc: {val_acc:.2f}%, F1: {val_f1_macro:.2f}%")
        print(classification_report(val_labels, val_preds, target_names=target_names))

        # 🔥 상세 Metric 파일 저장
        class_f1_str = "\t".join([f"{f1:.2f}" for f1 in val_f1_per_class])
        with open(metric_file, "a", encoding="utf-8-sig") as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{train_acc:.2f}\t{train_f1:.2f}\t{val_acc:.2f}\t{val_f1_macro:.2f}\t{class_f1_str}\n")

        # Early Stopping
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            counter = 0
            torch.save(model.state_dict(), "bert-base-lora-supcon-best.pth")
            print(f"🥇 Best Model Saved! (F1: {val_f1_macro:.2f}%)")
        else:
            counter += 1
            if counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}.")
                break

if __name__ == "__main__":
    train()