import pandas as pd
import openai
import json
import time
from tqdm import tqdm
import config
import utils

# API 키 설정
openai.api_key = utils.OPENAI_API_KEY

# --- 시스템 프롬프트 ---
PROMPT = """
[System Role]
You are an expert MBTI linguistic judge.

[Task]
You are provided with two text candidates: "Input" and "Output".
Your goal is to determine which text better reflects the specific [Target MBTI] traits.

[Definitions]
- Input: Candidate Text Option A
- Output: Candidate Text Option B
- Targets: The specific MBTI traits to evaluate against (e.g., Target IE: "E").

[Instructions]
1. Analyze the tone, vocabulary, and vibe of both texts.
2. Compare them strictly against the provided [Targets].
3. Select the candidate ("Input" or "Output") that aligns closer to the Target.

[Output Format]
Return ONLY a JSON object with keys: "winner_ie", "winner_ns", "winner_tf".
Values must be strictly "Input" or "Output".
"""

def get_judgement(text_input, text_output, target_ie, target_ns, target_tf):
# 두 문장을 비교하여 더 Target에 가까운 쪽(Input/Output)을 JSON으로 반환
    user_content = f"""
    [Targets]
    1. IE Target: {target_ie}
    2. NS Target: {target_ns}
    3. TF Target: {target_tf}

    [Input]
    {text_input}

    [Output]
    {text_output}
    """

    try:
        response = openai.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        # 에러 발생 시 None 반환
        return None

def main():
    # 1. 데이터 로드
    try:
        df = pd.read_csv(config.INPUT_FILE)
        print(f">> 데이터 로드 완료: {len(df)}개 샘플")
    except Exception as e:
        print(f">> 오류: {config.INPUT_FILE} 읽기 실패. {e}")
        return

    # 2. 투표 결과를 담을 리스트들
    votes_output_ie = []
    votes_output_ns = []
    votes_output_tf = []
    valid_vote_counts = []

    print(f">> SxS 반복 투표 평가 시작 (모델: {config.MODEL_NAME}, 반복: {config.VOTE_COUNT}회)...")

    # 3. 평가 루프 실행
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        time.sleep(config.SLEEP_TIME)
        
        input_text = str(row['input'])
        output_text = str(row['output'])
        t_ie = str(row['target_ie'])
        t_ns = str(row['target_ns'])
        t_tf = str(row['target_tf'])

        count_ie = 0
        count_ns = 0
        count_tf = 0
        success_calls = 0

        for _ in range(config.VOTE_COUNT):
            result = get_judgement(input_text, output_text, t_ie, t_ns, t_tf)
            
            if result:
                success_calls += 1
                if result.get('winner_ie') == 'Output': count_ie += 1
                if result.get('winner_ns') == 'Output': count_ns += 1
                if result.get('winner_tf') == 'Output': count_tf += 1
        
        votes_output_ie.append(count_ie)
        votes_output_ns.append(count_ns)
        votes_output_tf.append(count_tf)
        valid_vote_counts.append(success_calls)

    # 4. 결과 저장
    df['votes_output_ie'] = votes_output_ie
    df['votes_output_ns'] = votes_output_ns
    df['votes_output_tf'] = votes_output_tf
    df['valid_vote_count'] = valid_vote_counts

    df.to_csv(config.OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n>> 평가 완료! 결과 파일: {config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()