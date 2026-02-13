import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
from tqdm import tqdm

# --- 1. 모델 구조 정의 (동일 유지) ---
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

# --- 2. 테스트 데이터셋 정의 (동일 유지) ---
class TestTokenDataset(Dataset):
    def __init__(self, texts, tokenizer_name):
        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        sentence = self.texts[idx]
        tokens = self.tokenizer(
            str(sentence), return_tensors='pt', truncation=True,
            padding='max_length', max_length=512, add_special_tokens=True
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'token_type_ids': torch.zeros(512, dtype=torch.long),
        }

# --- 3. 메인 추론 함수 ---
def run_inference_to_index():
    CHECKPOINT_NAME = "monologg/kobigbird-bert-base"
    MODEL_PATH = '/home/summer24/DataFrom101/ddd/DKTC_classification/태경/best_f1_model.pth'
    TEST_JSON_PATH = 'processed_data/test.json'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 모델 로드
    print("🔄 모델 로딩 중...")
    model = CustomBertSupConModel(CHECKPOINT_NAME).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. 데이터 로드
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_json = json.load(f)
    
    test_ids = list(test_json.keys())
    test_texts = [test_json[tid]['text'] for tid in test_ids]
    
    test_loader = DataLoader(TestTokenDataset(test_texts, CHECKPOINT_NAME), batch_size=64, shuffle=False)

    # 3. 추론 (Inference)
    raw_preds = []
    with torch.no_grad():
        for inputs in tqdm(test_loader, desc="Inference"):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            logits, _ = model(**inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            raw_preds.extend(preds)

    # 4. 🔥 인덱스 재매핑 (가장 중요한 부분)
    # 모델 출력값(가나다순) -> 최종 제출 인덱스
    # 모델 0(갈취) -> 제출 1
    # 모델 1(기타) -> 제출 3
    # 모델 2(일반) -> 제출 4
    # 모델 3(직장) -> 제출 2
    # 모델 4(협박) -> 제출 0
    remapping_dict = {
        0: 1,  # 갈취
        1: 3,  # 기타 괴롭힘
        2: 4,  # 일반
        3: 2,  # 직장 내 괴롭힘
        4: 0   # 협박
    }
    
    final_preds = [remapping_dict[p] for p in raw_preds]

    # 5. 결과 저장
    submission_df = pd.DataFrame({
        'file_name': test_ids,
        'class': final_preds
    })

    output_filename = "submission_final_kobigbir.csv"
    submission_df.to_csv(output_filename, index=False, encoding='utf-8')
    
    print("-" * 30)
    print(f"✅ 재매핑 완료! 제출 파일 생성됨: {output_filename}")
    print(f"📊 매핑 결과 예시 (처음 5개):")
    for i in range(5):
        print(f"ID: {test_ids[i]} | 모델출력: {raw_preds[i]} -> 최종인덱스: {final_preds[i]}")

if __name__ == "__main__":
    run_inference_to_index()