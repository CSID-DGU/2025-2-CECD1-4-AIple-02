# 프로젝트 설정 파일

# --- 1. 모델 및 API 설정 ---
MODEL_NAME = "gpt-5-nano"
SLEEP_TIME = 0.0

# --- 2. 평가 전략 설정 ---
TRIAL_COUNT = 10
FEWSHOT_COUNT = 10

# --- 3. 파일 경로 설정 ---
# 정답지
GROUND_TRUTH_FILE = "ground_truth.csv"

# 평가 대상 파일
AGGREGATED_FILE = "evaluation_set_persona.csv"

# 최종 결과 파일
OUTPUT_FILE = "results_persona.csv"