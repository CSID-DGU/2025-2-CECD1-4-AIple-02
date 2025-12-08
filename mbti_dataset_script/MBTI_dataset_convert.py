import os, json, time, csv
import openai

MODEL = "gpt-5-mini"
REQUESTS_PER_MIN = 120
SLEEP_BETWEEN = 60.0 / REQUESTS_PER_MIN

PROMPT = """You are a style transfer system.

Goal:
Rewrite the sentence into a plain, everyday speaking tone that sounds neutral and low-personality. It should keep the same meaning but remove strong emotion, personal intensity, dramatic tone, confident judgments, or strong personality signals. The result should sound casual and human, but flat and understated.

Tone rules:
- Keep the meaning and level of detail similar.
- Use simple, everyday phrasing that sounds like normal conversation.
- Maintain casual tone, not formal and not robotic.
- Avoid strong emotion, extreme certainty, dramatic phrasing, or expressive emphasis.
- If the original has a judgment or negative tone, soften it to a mild observation (e.g., “it came across that way,” “it seemed like,” “I reacted to it”).
- Avoid identity statements ("I'm the type who..."), preferences, or personality cues.
- Maintain sentence length within ±30%.
- Output exactly one plain, low-expression, natural-sounding sentence.

Input:
{sentence}

Output format:
Return only a JSON object:
{{
  "neutral": "<rewritten sentence>"
}}
"""

def neutralize(sentence: str) -> str:
    prompt = PROMPT.format(sentence=sentence)
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role":"user","content":prompt}],
        )
    except Exception as e:
        print("[API ERROR]", type(e).__name__, str(e))
        raise

    cont = resp["choices"][0]["message"]["content"].strip()
    try:
        data = json.loads(cont)
        return data.get("neutral","").strip() or sentence
    except Exception as e:
        print("[PARSE WARN] raw:", cont[:200])
        #JSON 파싱 실패 시 따옴표 제거 후 첫 줄 사용
        line = cont.splitlines()[0].strip().strip('"')
        return line if line else sentence
    
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_csv(path, rows):
    fieldnames = ['Movie', 'Character', 'Input', 'Output', 'I-E', 'N-S', 'T-F']
    
    with open(path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

#데이터 쌍 구축 함수
def run_neutralization(input_jsonl, output_csv, api_key):
    openai.api_key = api_key
    
    #입력 파일 확인
    if not os.path.exists(input_jsonl):
        print(f"[Error] 파일을 찾을 수 없습니다: {input_jsonl}")
        return

    #라인 수 계산
    with open(input_jsonl, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
        
    results = []
    backoff = 1.0

    print(f"[Step 3] 스타일 변환 시작 (GPT): {input_jsonl} -> {total_lines} lines")

    for idx, rec in enumerate(read_jsonl(input_jsonl), 1):
        original = rec.get("text","").strip()
        label = rec.get("label","").strip()

        if not original or not label:
            continue

        neutral_text = original  #기본값
        
        #API 호출 재시도 로직
        for attempt in range(5):
            try:
                neutral_text = neutralize(original)
                time.sleep(SLEEP_BETWEEN)
                backoff = 1.0
                break
            except openai.error.RateLimitError:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except Exception:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

        #라벨 파싱 (I-E, N-S, T-F)
        ie = label[0] if len(label) > 0 else ""
        ns = label[1] if len(label) > 1 else ""
        tf = label[2] if len(label) > 2 else ""
        
        #Character 이름 생성
        character_val = f"{label[:3]}x Person" if len(label) >= 3 else "Unknown Person"

        row = {
            'Movie': 'Life',
            'Character': character_val,
            'Input': neutral_text,        #중립 문장
            'Output': original,           #성격 문장
            'I-E': ie,
            'N-S': ns,
            'T-F': tf
        }
        results.append(row)

        #진행률 표시
        pct = (idx / total_lines) * 100
        print(f"\r  Processed {idx}/{total_lines} ({pct:.1f}%)", end="")

    #결과 정렬 및 저장
    results.sort(key=lambda x: x["Character"])
    write_csv(output_csv, results)
    print(f"\n[Step 3] 변환 완료 -> {output_csv}")

# 단독 테스트 용 메인 함수
if __name__ == "__main__":
    TEST_API_KEY = ""
    IN_FILE = r""
    OUT_FILE = r""
    # run_neutralization(IN_FILE, OUT_FILE, TEST_API_KEY)