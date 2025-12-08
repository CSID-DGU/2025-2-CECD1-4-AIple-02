import pandas as pd
import openai
import json
import time
from tqdm import tqdm

import config
import utils

# --- 1. API 키 설정 ---
openai.api_key = utils.OPENAI_API_KEY
if not openai.api_key or openai.api_key == "sk-YOUR_API_KEY_GOES_HERE":
    print(">> 오류: utils.py 파일에 OpenAI API 키를 설정해야 합니다.")
    exit()

# --- 2. LLM에게 보낼 영문 시스템 프롬프트 ---
ENGLISH_SYSTEM_PROMPT = """
[System Role]
You are an linguistic analysis module specializing in the Myers-Briggs Type Indicator (MBTI).

[Task]
Analyze the tone and expression of the given text to perform a binary classification:
Does the text align more with 'E (Extraversion)' or 'I (Introversion)'?

[Instructions]
1.  This classification must be based solely on your internal, pre-trained knowledge of the MBTI E/I spectrum.
2.  Do not follow any external examples or subjective criteria. Your judgment must remain 'pure'.
3.  Do not include any conversational elements, explanations, or greetings.
4.  Your response must be only a valid JSON object.

[Output Format Specification]
- The response must be a valid JSON object.
- The JSON object must contain exactly two keys:
  1.  `label`: The classification result. Must be the string "E" or "I".
  2.  `confidence`: Your confidence in this classification (a float between 0.0 and 1.0).
"""

# --- 3. LLM API 호출 함수 ---
def get_llm_judgment(text_to_classify):
    try:
        response = openai.chat.completions.create(
            model=config.MODEL_NAME, # config.py에서 모델 이름 사용
            messages=[
                {"role": "system", "content": ENGLISH_SYSTEM_PROMPT},
                {"role": "user", "content": text_to_classify}
            ],
            response_format={"type": "json_object"},
            # temperature=0.0 
        )
        json_response = json.loads(response.choices[0].message.content)
        return json_response.get('label'), json_response.get('confidence')
    
    except json.JSONDecodeError:
        print(f"\n>> 오류: JSON 파싱 실패. LLM 응답: {response.choices[0].message.content}")
        return "PARSE_ERROR", None
    except Exception as e:
        print(f"\n>> 오류: API 호출 실패. 문장: {text_to_classify}, 에러: {e}")
        return "API_ERROR", None

# --- 4. 메인 실행 함수 ---
def main():
    try:
        # config.py에서 파일 이름 사용
        df = pd.read_csv(config.INPUT_FILE, encoding='utf-8')
    except FileNotFoundError:
        print(f">> 오류: '{config.INPUT_FILE}'을 찾을 수 없습니다. 파일 경로와 이름을 확인하세요.")
        return
    except UnicodeDecodeError:
        print(f">> 오류: '{config.INPUT_FILE}' 파일 인코딩 오류. 파일을 'utf-8'로 저장해 주세요.")
        return

    predicted_labels = []
    confidences = []

    print(f">> 총 {len(df)}개의 문장을 찾았습니다. LLM 심판 분류를 시작합니다...")
    print(f">> 모델: {config.MODEL_NAME}, 입력: {config.INPUT_FILE}")

    # tqdm으로 진행 상황 표시
    for text in tqdm(df['text']):
        # config.py에서 대기 시간 사용
        time.sleep(config.SLEEP_TIME) 
        
        label, conf = get_llm_judgment(str(text)) 
        
        predicted_labels.append(label)
        confidences.append(conf)

    df['predicted_label'] = predicted_labels
    df['confidence'] = confidences

    # config.py에서 출력 파일 이름 사용
    df.to_csv(config.OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n>> 작업 완료! 예측 결과가 {config.OUTPUT_FILE}에 저장되었습니다.")

if __name__ == "__main__":
    main()