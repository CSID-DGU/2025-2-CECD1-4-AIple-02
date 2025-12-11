import pandas as pd
import openai
import json
import time
from tqdm import tqdm
import config
import utils

openai.api_key = utils.OPENAI_API_KEY

# --- 프롬프트 ---
PROMPT = """
[System Role]
You are an MBTI linguistic analyst.

[Task]
Classify the text into three dimensions based on tone and word choice:
1. I (Introversion) vs E (Extraversion)
2. N (Intuition) vs S (Sensing)
3. T (Thinking) vs F (Feeling)

[Output Format]
Provide ONLY a JSON object with keys "ie", "ns", "tf".
Values must be the single uppercase letter ("I", "E", "N", "S", "T", "F").

[Example]
{"ie": "E", "ns": "N", "tf": "F"}
"""

def get_judgment(text):
    try:
        response = openai.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"},
            # temperature=0.0 # gpt-4o 등 최신 모델 호환성을 위해 주석 처리
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"\n>> Error: {e}")
        return None

def main():
    # 1. 데이터 로드
    try:
        df = pd.read_csv(config.INPUT_FILE)
    except Exception as e:
        print(f">> 오류: {config.INPUT_FILE} 읽기 실패. {e}")
        return

    print(f">> 3가지 속성(IE, NS, TF) 판정 시작... (모델: {config.MODEL_NAME})")

    # 2. 결과 담을 리스트
    preds_ie = []
    preds_ns = []
    preds_tf = []

    # 3. API 호출 루프
    for text in tqdm(df['text']):
        time.sleep(config.SLEEP_TIME)
        
        result = get_judgment(str(text))
        
        if result:
            preds_ie.append(result.get('ie'))
            preds_ns.append(result.get('ns'))
            preds_tf.append(result.get('tf'))
        else:
            preds_ie.append(None)
            preds_ns.append(None)
            preds_tf.append(None)

    # 4. 결과 저장
    df['pred_ie'] = preds_ie
    df['pred_ns'] = preds_ns
    df['pred_tf'] = preds_tf

    df.to_csv(config.OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n>> 완료! 결과가 {config.OUTPUT_FILE}에 저장되었습니다.")

if __name__ == "__main__":
    main()