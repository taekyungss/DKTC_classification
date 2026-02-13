import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, BertModel

# 0. Huggingface Metadata Error Bypass
try:
    import importlib.metadata
    importlib.metadata.version('huggingface-hub')
except Exception:
    def dummy_version(name): return "0.29.1"
    importlib.metadata.version = dummy_version

# --- 1. Class Definitions ---
class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained):
        self.data = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['conversation']
        label = self.data.iloc[idx]['label']
        tokens = self.tokenizer(str(sentence), return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'token_type_ids': torch.zeros(512, dtype=torch.long),
        }, torch.tensor(label, dtype=torch.long)

class CustomBertSupConModel(nn.Module):
    def __init__(self, bert_pretrained, dropout_rate=0.5):
        super(CustomBertSupConModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained)
        self.dr = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(768, 5)
        self.projection_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 128)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = output['last_hidden_state'][:, 0, :]
        logits = self.fc(self.dr(cls_token))
        proj_feat = F.normalize(self.projection_head(cls_token), dim=1)
        return logits, proj_feat

# --- 2. Analysis Functions ---
def run_full_analysis(model, val_loader, val_df, mapping, device, result_dir):
    model.eval()
    all_preds, all_labels, errors = [], [], []
    
    eng_mapping = {
        '협박 대화': 'Threat', '갈취 대화': 'Extortion',
        '직장 내 괴롭힘 대화': 'Workplace_Bullying',
        '기타 괴롭힘 대화': 'Other_Bullying', '일반 대화': 'Normal'
    }
    
    print("🚀 1/2: Starting Inference & Error Analysis...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            # [수정] SupCon 모델은 (logits, proj_feat)를 반환하므로 unpack 처리
            logits, _ = model(**inputs) 
            preds = logits.argmax(dim=1)

            wrong_idx = torch.where(preds != labels)[0]
            for idx in wrong_idx:
                abs_idx = i * val_loader.batch_size + idx.item() 
                true_name = mapping[labels[idx].item()]
                pred_name = mapping[preds[idx].item()]
                
                true_eng = eng_mapping.get(true_name, true_name)
                pred_eng = eng_mapping.get(pred_name, pred_name)
                
                errors.append({
                    "conversation": val_df.iloc[abs_idx]['conversation'],
                    "true_label": true_name,
                    "pred_label": pred_name,
                    "error_pair": f"{true_eng} -> {pred_eng}"
                })
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    error_df = pd.DataFrame(errors)
    error_df.to_csv(f"{result_dir}/inference_error_analysis_supcon.csv", index=False, encoding='utf-8-sig')
    
    error_counts = error_df['error_pair'].value_counts().reset_index()
    error_counts.columns = ['Error_Pair', 'Count']
    
    target_names_eng = [eng_mapping.get(mapping[i], mapping[i]) for i in range(len(mapping))]

    # Plot 1: Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=target_names_eng, yticklabels=target_names_eng)
    plt.title('Confusion Matrix (SupCon)'); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.savefig(f"{result_dir}/confusion_matrix_supcon.png"); plt.close()

    print(f"✅ Analysis reports saved to ({result_dir})")

def plot_tsne_visualization(model, data_loader, device, mapping, result_dir):
    model.eval()
    features, labels = [], []
    eng_mapping = {
        '협박 대화': 'Threat', '갈취 대화': 'Extortion',
        '직장 내 괴롭힘 대화': 'Workplace_Bullying',
        '기타 괴롭힘 대화': 'Other_Bullying', '일반 대화': 'Normal'
    }
    
    print("🚀 2/2: Starting t-SNE Feature Extraction...")
    with torch.no_grad():
        for inputs, lb in data_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # [CLS] 토큰 임베딩 추출
            output = model.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
            features.append(output['last_hidden_state'][:, 0, :].cpu().numpy())
            labels.append(lb.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    eng_label_list = [eng_mapping.get(mapping[l], mapping[l]) for l in labels]
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=eng_label_list, palette='hls', alpha=0.6)
    plt.title('SupCon BERT Embeddings t-SNE Visualization')
    plt.savefig(f"{result_dir}/tsne_plot_supcon.png"); plt.close()
    print(f"✅ t-SNE plot saved to ({result_dir}/tsne_plot_supcon.png)")

# --- 3. Main Execution ---
if __name__ == "__main__":
    CHECKPOINT_NAME = "klue/bert-base"
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = '/home/summer24/DataFrom101/ddd/DKTC_classification/태경/bert-base-lora-supcon-best.pth'
    RESULT_DIR = './analysis_results_lora'
    os.makedirs(RESULT_DIR, exist_ok=True)
    plt.rcParams['axes.unicode_minus'] = False 

    # 데이터 로드
    df = pd.read_csv("/home/summer24/DataFrom101/ddd/DKTC_classification/태경/processed_data/train_combined_no_sep.csv")
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['class'])
    mapping = dict(zip(range(len(le.classes_)), le.classes_))
    # _, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    val_loader = DataLoader(TokenDataset(df, CHECKPOINT_NAME), batch_size=128, shuffle=False)
    
    # 모델 로드 (SupCon 구조)
    model = CustomBertSupConModel(CHECKPOINT_NAME)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    run_full_analysis(model, val_loader, val_df, mapping, DEVICE, RESULT_DIR)
    plot_tsne_visualization(model, val_loader, DEVICE, mapping, RESULT_DIR)
    
    print(f"\n✨ Analysis complete! Check the '{RESULT_DIR}' folder.")