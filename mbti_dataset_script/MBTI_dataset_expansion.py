import pandas as pd
import json
import time
import re
import openai
from typing import List


MODEL = "gpt-5-mini"
REQUESTS_PER_MIN = 50
SLEEP_BETWEEN = 60.0 / REQUESTS_PER_MIN
END_PUNCT = re.compile(r'[.?!]$')

def build_prompt(label: str, text: str) -> str:
    prompt = f"""
You are a text analysis expert.
The following text was written by a person with the MBTI type {label}.

<Task>
Select 5 sentences that best represent typical traits of the {label} personality type.

<Rules>
1. Only choose complete, meaningful sentences.
A sentence must:
express a full, coherent idea
not be cut off or abruptly end
end with "., ?, or !"

2. contain at least 6 words

3. Return sentences exactly as they appear in the original text.

4. Do not paraphrase, modify, merge, or fabricate sentences.

5. If fewer than 5 valid sentences exist, return only the valid ones.

<Output strictly as JSON>
Return ONLY a JSON object with this exact shape:
{{
  "sentences": ["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"]
}}
- Do not include any commentary or extra keys.
- If fewer than 5, include fewer items in the array.

Text:
{text}
""".strip()
    return prompt

def call_gpt(prompt: str) -> List[str]:
    #호환성을 위해 기존 방식 유지
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp["choices"][0]["message"]["content"].strip()
        
        #JSON 파싱
        data = json.loads(content)
        sentences = data.get("sentences", [])
        if not isinstance(sentences, list):
            raise ValueError("`sentences` is not a list.")
        return [s.strip() for s in sentences if isinstance(s, str)]
        
    except Exception as e: 
        #에러가 발생하면 콘솔에 출력
        print(f"  [API Error in call_gpt] {type(e).__name__}: {e}")
        return []

def is_valid_sentence(s: str) -> bool:
    return (
        isinstance(s, str)
        and len(s.split()) >= 6
        and END_PUNCT.search(s) is not None
    )
    
def expand_row(text: str, label: str) -> List[dict]:
    prompt = build_prompt(label, text)
    sentences = call_gpt(prompt)
    
    cleaned = []
    for s in sentences:
        s_clean = s.strip()
        #원문에 있는 내용인지 한 번 더 체크
        if is_valid_sentence(s_clean) and s_clean in text:
            cleaned.append({"text": s_clean, "label": label})
    return cleaned[:5]

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def write_jsonl(path: str, rows: List[dict], mode: str="a"):
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

#핵심 문장 추출 함수
def run_expansion(input_jsonl, output_jsonl, api_key):
    openai.api_key = api_key
    
    print(f"[Step 2] 문장 추출 시작 (GPT): {input_jsonl}")

    #출력 파일 초기화
    open(output_jsonl, "w", encoding="utf-8").close()

    backoff = 1.0
    
    #총 라인 수 계산 (진행률 표시 위함, 필요시 사용)
    #total_lines = sum(1 for _ in open(input_jsonl, "r", encoding="utf-8"))
    
    for idx, rec in enumerate(read_jsonl(input_jsonl), 1):
        text  = rec.get("text", "")
        label = rec.get("label", "")

        if not text or not label:
            continue

        #API 호출 + 재시도 로직
        for attempt in range(5):
            try:
                rows = expand_row(text, label)
                if rows:
                    write_jsonl(output_jsonl, rows, mode="a")
                
                #속도 제한 준수
                time.sleep(SLEEP_BETWEEN)
                backoff = 1.0
                break
            except openai.error.RateLimitError:
                print(f"  [Rate Limit] {backoff}초 대기...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except Exception as e:
                print(f"  [Error] {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                
        #진행 상황 로그
        if idx % 10 == 0:
            print(f"  Processed {idx} lines...", end='\r')

    print(f"\n[Step 2] 문장 추출 완료 -> {output_jsonl}")

#단독 테스트용 메인 함수
if __name__ == "__main__":
    #테스트 시에는 아래 변수를 본인 환경에 맞게 수정해서 실행
    TEST_API_KEY = "" 
    IN_FILE = r""
    OUT_FILE = r""
    
    #run_expansion(IN_FILE, OUT_FILE, TEST_API_KEY)