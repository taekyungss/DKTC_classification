import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel

# -------------------------------------------------
# 0. Huggingface Metadata Error Bypass
# -------------------------------------------------
try:
    import importlib.metadata
    importlib.metadata.version('huggingface-hub')
except Exception:
    def dummy_version(name): return "0.29.1"
    importlib.metadata.version = dummy_version

# -------------------------------------------------
# 1. SupCon Loss
# -------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-6)
        return -mean_log_prob_pos.mean()

# -------------------------------------------------
# 2. Dataset
# -------------------------------------------------
class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_name):
        self.data = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['conversation']
        label = self.data.iloc[idx]['label']

        tokens = self.tokenizer(
            str(sentence),
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'token_type_ids': torch.zeros(512, dtype=torch.long)
        }, torch.tensor(label, dtype=torch.long)

# -------------------------------------------------
# 3. Model
# -------------------------------------------------
class CustomBertSupConModel(nn.Module):
    def __init__(self, bert_pretrained, dropout_rate=0.5):
        super(CustomBertSupConModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained)
        self.dr = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(768, 5)

        self.projection_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 128)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls = output.last_hidden_state[:, 0, :]
        logits = self.fc(self.dr(cls))
        proj_feat = F.normalize(self.projection_head(cls), dim=1)

        return logits, proj_feat

# -------------------------------------------------
# 4. Train / Eval
# -------------------------------------------------
def train_epoch(model, loader, ce_loss_fn, sc_loss_fn, optimizer, device, alpha=0.7):
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, proj_feat = model(**inputs)

        loss_ce = ce_loss_fn(logits, labels)
        loss_sc = sc_loss_fn(proj_feat, labels)
        loss = alpha * loss_ce + (1 - alpha) * loss_sc

        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        f1_score(all_labels, all_preds, average='weighted'),
        f1_score(all_labels, all_preds, average=None)
    )

def evaluate(model, loader, ce_loss_fn, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            logits, _ = model(**inputs)
            loss = ce_loss_fn(logits, labels)

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        f1_score(all_labels, all_preds, average='weighted'),
        f1_score(all_labels, all_preds, average=None)
    )

# -------------------------------------------------
# 5. Main
# -------------------------------------------------
if __name__ == "__main__":

    CHECKPOINT_NAME = "monologg/kobigbird-bert-base"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_SAVE_PATH = "best_f1_model.pth"
    METRIC_PATH = "training_metrics.txt"

    TARGET_NAMES = ['Threat','Extortion','Workplace_Bullying','Other_Bullying','Normal']

    df = pd.read_csv("processed_data/train_combined_no_sep.csv")

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["class"])

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    class_weights = torch.tensor([1.6,1.1,0.6,0.9,1.4]).to(DEVICE)

    train_dataset = TokenDataset(train_df, CHECKPOINT_NAME)
    val_dataset = TokenDataset(val_df, CHECKPOINT_NAME)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = CustomBertSupConModel(CHECKPOINT_NAME).to(DEVICE)

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    sc_loss_fn = SupConLoss(temperature=0.07)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    EPOCHS = 100
    patience = 10
    counter = 0
    best_f1 = 0.0

    for epoch in range(EPOCHS):

        t_loss, t_acc, t_f1, _ = train_epoch(
            model, train_loader,
            ce_loss_fn, sc_loss_fn,
            optimizer, DEVICE
        )

        v_loss, v_acc, v_f1, v_cls_f1 = evaluate(
            model, val_loader,
            ce_loss_fn, DEVICE
        )

        print(f"[{epoch+1}/{EPOCHS}] "
              f"Train F1: {t_f1:.4f} | "
              f"Val F1: {v_f1:.4f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            counter = 0
            print(f"🔥 Best Model Saved! (Val F1 = {best_f1:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print("🛑 Early Stopping")
                break

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print("✅ Best F1 Model Loaded")
