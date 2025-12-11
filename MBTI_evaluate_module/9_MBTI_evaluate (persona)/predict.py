import pandas as pd
import openai
import json
import time
import random
from tqdm import tqdm
import config
import utils

openai.api_key = utils.OPENAI_API_KEY

# --- 데이터 로더 (정답지) ---
def load_ground_truth_examples():
    try:
        df = pd.read_csv(config.GROUND_TRUTH_FILE)
        examples = {}
        target_col = 'output'
        if target_col not in df.columns:
             for c in df.columns:
                 if 'output' in c.lower() or 'generated output' in c.lower(): target_col = c; break
        
        for _, row in df.iterrows():
            text = str(row.get(target_col, "")).strip()
            if not text: continue
            
            # MBTI 컬럼 확인
            mbti_col = next((c for c in df.columns if c.lower() == 'mbti'), None)
            if mbti_col and pd.notna(row[mbti_col]):
                m = str(row[mbti_col]).strip().upper()
                if m not in examples: examples[m] = []
                examples[m].append(text)
        return examples
    except Exception as e:
        return {}

GT_EXAMPLES = load_ground_truth_examples()

def get_fewshot_prompt(target_trait):
    if target_trait not in GT_EXAMPLES: return "(No examples available)"
    pool = GT_EXAMPLES[target_trait]
    samples = random.sample(pool, min(config.FEWSHOT_COUNT, len(pool)))
    text = ""
    for i, s in enumerate(samples, 1): text += f"- Reference Style {i}: \"{s}\"\n"
    return text

# --- 프롬프트 ---
PROMPT_TEMPLATE = """
[System Role]
You are an expert MBTI linguistic judge.
You have been provided with 'Ground Truth' examples that perfectly embody specific MBTI traits.

[Task]
Compare two text collections ("Input" vs "Output") representing a character's speech patterns.
Determine which collection better reflects the [Target MBTI].
Crucial: Your judgment must be based on the style, tone, and vibe of the provided [Reference Examples].

[Reference Examples for Target: {target}]
{examples}

[Instructions]
1. Read the Reference Examples carefully. This is the "Gold Standard".
2. Compare the 'Input' and 'Output' text collections against these examples.
3. Select the collection that aligns closer to the Reference Examples in terms of overall persona and style.

[Output Format]
Return ONLY a JSON object.
Key: "{key}"
Value: "Input" or "Output".
"""

def get_judgement(text_input, text_output, target, key):
    examples_text = get_fewshot_prompt(target)
    system_prompt = PROMPT_TEMPLATE.format(target=target, examples=examples_text, key=key)
    
    # 텍스트가 너무 길 경우를 대비해 앞부분 30000자만 사용 (필요시 조정)
    safe_input = text_input[:30000]
    safe_output = text_output[:30000]

    user_content = f"""
    [Target to Evaluate]
    {target} ({key.replace('winner_', '').upper()} dimension)

    [Input Collection]
    {safe_input}

    [Output Collection]
    {safe_output}
    """

    try:
        response = openai.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return None

def main():
    try:
        # 캐릭터별로 묶인 파일 로드
        df = pd.read_csv(config.AGGREGATED_FILE)
        print(f">> 캐릭터별 데이터 로드 완료: {len(df)}명")
    except Exception as e:
        print(f">> 오류: {config.AGGREGATED_FILE} 읽기 실패. {e}")
        return

    votes_ie, votes_ns, votes_tf = [], [], []
    valid_cnts = []

    print(f">> Persona 평가 시작 (N={config.TRIAL_COUNT})...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        time.sleep(config.SLEEP_TIME)
        
        inp = str(row['input'])
        out = str(row['output'])
        tie = row['target_ie'] if pd.notna(row['target_ie']) else None
        tns = row['target_ns'] if pd.notna(row['target_ns']) else None
        ttf = row['target_tf'] if pd.notna(row['target_tf']) else None

        c_ie, c_ns, c_tf = 0, 0, 0
        calls = 0

        for _ in range(config.TRIAL_COUNT):
            if tie:
                res = get_judgement(inp, out, tie, 'winner_ie')
                if res:
                    calls = max(calls, 1)
                    if res.get('winner_ie') == 'Output': c_ie += 1
            if tns:
                res = get_judgement(inp, out, tns, 'winner_ns')
                if res:
                    calls = max(calls, 1)
                    if res.get('winner_ns') == 'Input': c_ns += 1
            if ttf:
                res = get_judgement(inp, out, ttf, 'winner_tf')
                if res:
                    calls = max(calls, 1)
                    if res.get('winner_tf') == 'Output': c_tf += 1
        
        votes_ie.append(c_ie)
        votes_ns.append(c_ns)
        votes_tf.append(c_tf)
        valid_cnts.append(config.TRIAL_COUNT)

    df['votes_output_ie'] = votes_ie
    df['votes_output_ns'] = votes_ns
    df['votes_output_tf'] = votes_tf
    df['valid_trial_count'] = valid_cnts

    df.to_csv(config.OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n>> 완료! {config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()