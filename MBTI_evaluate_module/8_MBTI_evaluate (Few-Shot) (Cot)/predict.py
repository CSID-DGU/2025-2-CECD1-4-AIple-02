import pandas as pd
import openai
import json
import time
import random
from tqdm import tqdm
import config
import utils

openai.api_key = utils.OPENAI_API_KEY

def load_ground_truth_examples():
    try:
        df = pd.read_csv(config.GROUND_TRUTH_FILE)
        examples = {}
        target_col = 'output' 
        for _, row in df.iterrows():
            text = str(row.get(target_col, "")).strip()
            if not text: continue
            targets = [row.get('target_ie'), row.get('target_ns'), row.get('target_tf')]
            if 'MBTI' in row: targets.append(row['MBTI'])
            for t in targets:
                if pd.notna(t):
                    t = str(t).strip().upper()
                    if t not in examples: examples[t] = []
                    examples[t].append(text)
        return examples
    except Exception as e:
        return {}

GT_EXAMPLES = load_ground_truth_examples()

def get_fewshot_prompt(target_trait):
    if target_trait not in GT_EXAMPLES:
        return "(No examples available)"
    pool = GT_EXAMPLES[target_trait]
    samples = random.sample(pool, min(config.FEWSHOT_COUNT, len(pool)))
    text = ""
    for i, s in enumerate(samples, 1):
        text += f"- Reference Style {i}: \"{s}\"\n"
    return text

# --- 프롬프트 ---
PROMPT_TEMPLATE = """
[System Role]
You are an expert MBTI linguistic judge.

[Task]
Compare two text candidates ("Input" vs "Output") and determine which one better reflects the [Target MBTI].
Your judgment must be grounded in the style of the provided [Reference Examples].

[Reference Examples for Target: {target}]
{examples}

[Instructions]
1. Read the [Reference Examples] carefully to understand the exact tone, vocabulary, and vibe of the target MBTI.
2. Step-by-Step Reasoning:
   - Evaluate 'Input': How similar is its style to the Reference Examples?
   - Evaluate 'Output': How similar is its style to the Reference Examples?
   - Compare: Which of the two candidates bears a stronger resemblance to the Reference Examples?
3. Select the winner.
   - If 'Output' is closer to the Reference: Select "Output".
   - If 'Input' is closer to the Reference: Select "Input".

[Output Format]
Return a JSON object with two keys:
1. "reasoning": A short explanation of your comparison (1 sentence).
2. "{key}": "Input" or "Output".
"""

def get_judgement(text_input, text_output, target, key):
    examples_text = get_fewshot_prompt(target)
    system_prompt = PROMPT_TEMPLATE.format(target=target, examples=examples_text, key=key)
    
    user_content = f"""
    [Target to Evaluate]
    {target} ({key.replace('winner_', '').upper()} dimension)

    [Input]
    {text_input}

    [Output]
    {text_output}
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
        df = pd.read_csv(config.INPUT_FILE)
        print(f">> 평가 대상 로드 완료: {len(df)}개")
    except Exception as e:
        print(f">> 오류: {config.INPUT_FILE} 읽기 실패. {e}")
        return

    votes_ie, votes_ns, votes_tf = [], [], []
    reasons = [] 
    valid_cnts = []

    print(f">> CoT 강화 평가 시작 (N={config.TRIAL_COUNT})...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        time.sleep(config.SLEEP_TIME)
        
        inp = str(row['input'])
        out = str(row['output'])
        
        tie = row['target_ie'] if pd.notna(row['target_ie']) else None
        tns = row['target_ns'] if pd.notna(row['target_ns']) else None
        ttf = row['target_tf'] if pd.notna(row['target_tf']) else None

        c_ie, c_ns, c_tf = 0, 0, 0
        calls = 0
        last_reason = "" # 마지막 판단의 이유 저장

        for _ in range(config.TRIAL_COUNT):
            if tie:
                res = get_judgement(inp, out, tie, 'winner_ie')
                if res:
                    calls = max(calls, 1)
                    if res.get('winner_ie') == 'Output': c_ie += 1
                    last_reason = res.get('reasoning', '')
            
            if tns:
                res = get_judgement(inp, out, tns, 'winner_ns')
                if res:
                    calls = max(calls, 1)
                    if res.get('winner_ns') == 'Output': c_ns += 1
                    last_reason = res.get('reasoning', '')

            if ttf:
                res = get_judgement(inp, out, ttf, 'winner_tf')
                if res:
                    calls = max(calls, 1)
                    if res.get('winner_tf') == 'Output': c_tf += 1
                    last_reason = res.get('reasoning', '')
        
        votes_ie.append(c_ie)
        votes_ns.append(c_ns)
        votes_tf.append(c_tf)
        reasons.append(last_reason)
        valid_cnts.append(config.TRIAL_COUNT)

    df['votes_output_ie'] = votes_ie
    df['votes_output_ns'] = votes_ns
    df['votes_output_tf'] = votes_tf
    df['valid_trial_count'] = valid_cnts
    df['judge_reason'] = reasons

    df.to_csv(config.OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n>> 완료! {config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()