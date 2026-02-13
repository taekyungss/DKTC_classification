import pandas as pd
import os
from llama_cpp import Llama
from transformers import AutoTokenizer
from tqdm import tqdm

# 1. 모델 설정
model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Llama(
    model_path='/home/summer24/DataFrom101/ddd/DKTC_classification/태경/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf',
    n_ctx=2048
)

# 2. 파일 경로 설정
input_path = '/home/summer24/DataFrom101/ddd/DKTC_classification/태경/normal_conversation.csv'
output_path = '/home/summer24/DataFrom101/ddd/DKTC_classification/태경/normal_conversation_rephrased.csv'

df = pd.read_csv(input_path)

# 3. 변환 함수 (기존과 동일)
def rephrase_conversation(text):
    system_prompt = "너는 문장에서 이름과 호칭을 제거하고 자연스러운 구어체로 다듬는 전문가야."
    user_input = f"다음 대화에서 이름과 호칭(예: 현우야, 혜진아 등)을 삭제하고, 친근한 구어체로 재구성해줘:\n\n{text}"
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = model(prompt, max_tokens=512, stop=["<|eot_id|>"], temperature=0.7, top_p=0.9)
    return response['choices'][0]['text'].strip()

# 4. 루프를 돌며 한 줄씩 처리 및 즉시 저장
print(f"총 {len(df)}개의 데이터를 처리를 시작합니다.")

for i, row in tqdm(df.iterrows(), total=len(df)):
    # 변환 수행
    rephrased_text = rephrase_conversation(row['conversation'])
    
    # 결과 데이터 생성
    result_df = pd.DataFrame([{
        'idx': row['idx'],
        'class': row['class'],
        'conversation': rephrased_text
    }])
    
    # 파일이 없으면 헤더를 포함해 저장, 있으면 내용만 추가(append)
    if not os.path.exists(output_path):
        result_df.to_csv(output_path, index=False, mode='w', encoding='utf-8-sig')
    else:
        result_df.to_csv(output_path, index=False, mode='a', encoding='utf-8-sig', header=False)

print(f"모든 변환 작업이 완료되어 '{output_path}'에 저장되었습니다.")