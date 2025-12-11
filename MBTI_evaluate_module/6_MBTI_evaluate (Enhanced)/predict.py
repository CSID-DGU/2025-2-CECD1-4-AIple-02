import pandas as pd
import openai
import json
import time
from tqdm import tqdm
import config
import utils

openai.api_key = utils.OPENAI_API_KEY

# I vs E 프롬프트
PROMPT_IE = """
[System Role]
You are an expert MBTI linguistic judge specializing in the Introversion (I) vs. Extraversion (E) dimension.

[Task]
You are provided with two text candidates: "Input" and "Output".
Your goal is to determine which text better reflects the target trait (either I or E).

[MBTI Trait Definitions: I vs E]
Note: These are mutually exclusive opposites.
- I (Introversion): Internal energy, reflective, calm, reserved, depth-oriented.
- E (Extraversion): External energy, expressive, dynamic, social, breadth-oriented.

[Instructions]
1. Analyze the tone, vocabulary, and vibe of both texts specifically for I/E traits.
2. Compare them strictly against the provided [Target].
3. Since traits are binary, being closer to E means being further from I.
4. Select the candidate ("Input" or "Output") that aligns closer to the Target.

[Output Format]
Return ONLY a JSON object with the key: "winner_ie".
Value must be strictly "Input" or "Output".
"""

# N vs S 프롬프트
PROMPT_NS = """
[System Role]
You are an expert MBTI linguistic judge specializing in the Intuition (N) vs. Sensing (S) dimension.

[Task]
You are provided with two text candidates: "Input" and "Output".
Your goal is to determine which text better reflects the target trait (either N or S).

[MBTI Trait Definitions: N vs S]
Note: These are mutually exclusive opposites.
- N (Intuition): Abstract, conceptual, future-oriented, metaphorical, big picture.
- S (Sensing): Concrete, sensory, detail-oriented, literal, present reality.

[Instructions]
1. Analyze the tone, vocabulary, and vibe of both texts specifically for N/S traits.
2. Compare them strictly against the provided [Target].
3. Since traits are binary, being closer to N means being further from S.
4. Select the candidate ("Input" or "Output") that aligns closer to the Target.

[Output Format]
Return ONLY a JSON object with the key: "winner_ns".
Value must be strictly "Input" or "Output".
"""

# T vs F 프롬프트
PROMPT_TF = """
[System Role]
You are an expert MBTI linguistic judge specializing in the Thinking (T) vs. Feeling (F) dimension.

[Task]
You are provided with two text candidates: "Input" and "Output".
Your goal is to determine which text better reflects the target trait (either T or F).

[MBTI Trait Definitions: T vs F]
Note: These are mutually exclusive opposites.
- T (Thinking): Logical, objective, analytical, fact-based, critique-oriented.
- F (Feeling): Emotional, subjective, empathetic, value-based, harmony-oriented.

[Instructions]
1. Analyze the tone, vocabulary, and vibe of both texts specifically for T/F traits.
2. Compare them strictly against the provided [Target].
3. Since traits are binary, being closer to T means being further from F.
4. Select the candidate ("Input" or "Output") that aligns closer to the Target.

[Output Format]
Return ONLY a JSON object with the key: "winner_tf".
Value must be strictly "Input" or "Output".
"""

def get_judgement(text_input, text_output, target_trait, dimension):
    """
    dimension: 'IE', 'NS', or 'TF'
    target_trait: e.g., 'E', 'N', 'T'
    """
    if dimension == 'IE':
        system_prompt = PROMPT_IE
        target_desc = f"Target Trait: {target_trait} (Introversion vs Extraversion)"
    elif dimension == 'NS':
        system_prompt = PROMPT_NS
        target_desc = f"Target Trait: {target_trait} (Intuition vs Sensing)"
    elif dimension == 'TF':
        system_prompt = PROMPT_TF
        target_desc = f"Target Trait: {target_trait} (Thinking vs Feeling)"
    else:
        return None

    user_content = f"""
    [Target to Evaluate]
    {target_desc}

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
        print(f">> 데이터 로드 완료: {len(df)}개 샘플")
    except Exception as e:
        print(f">> 오류: {config.INPUT_FILE} 읽기 실패. {e}")
        return

    # 투표 결과를 담을 리스트들
    votes_output_ie = []
    votes_output_ns = []
    votes_output_tf = []
    valid_trial_counts = []

    print(f">> SxS 개별 프롬프트 평가 시작 (모델: {config.MODEL_NAME}, 반복: {config.TRIAL_COUNT}회)...")

    # 3. 평가 루프 실행
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        time.sleep(config.SLEEP_TIME)
        
        input_text = str(row['input'])
        output_text = str(row['output'])
        
        # 타겟 확인
        tie = row['target_ie'] if pd.notna(row['target_ie']) else None
        tns = row['target_ns'] if pd.notna(row['target_ns']) else None
        ttf = row['target_tf'] if pd.notna(row['target_tf']) else None

        count_ie = 0
        count_ns = 0
        count_tf = 0
        success_calls = 0 

        # IE 평가
        if tie:
            for _ in range(config.TRIAL_COUNT):
                res = get_judgement(input_text, output_text, tie, 'IE')
                if res and res.get('winner_ie') == 'Output':
                    count_ie += 1
        
        # NS 평가
        if tns:
            for _ in range(config.TRIAL_COUNT):
                res = get_judgement(input_text, output_text, tns, 'NS')
                if res and res.get('winner_ns') == 'Output':
                    count_ns += 1

        # TF 평가
        if ttf:
            for _ in range(config.TRIAL_COUNT):
                res = get_judgement(input_text, output_text, ttf, 'TF')
                if res and res.get('winner_tf') == 'Output':
                    count_tf += 1
        
        valid_trial_counts.append(config.TRIAL_COUNT) 
        
        votes_output_ie.append(count_ie)
        votes_output_ns.append(count_ns)
        votes_output_tf.append(count_tf)

    # 4. 결과 저장
    df['votes_output_ie'] = votes_output_ie
    df['votes_output_ns'] = votes_output_ns
    df['votes_output_tf'] = votes_output_tf
    df['valid_trial_count'] = valid_trial_counts

    df.to_csv(config.OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n>> 평가 완료! 결과 파일: {config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()