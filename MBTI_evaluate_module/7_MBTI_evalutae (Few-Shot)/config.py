# 프로젝트 설정 파일

# --- 1. 모델 및 API 설정 ---
# 사용 모델
MODEL_NAME = "gpt-5-nano"
SLEEP_TIME = 0.0

# --- 2. 평가 전략 설정 ---
# N=2 (1:1 동점 제외, 2:0 확정 승리만 인정)
TRIAL_COUNT = 1
FEWSHOT_COUNT = 10

# --- 3. 파일 경로 설정 ---
# 정답지
GROUND_TRUTH_FILE = "ground_truth.csv"

# 평가할 대상 파일
INPUT_FILE = "evaluation_set.csv"

# 결과 저장 파일
OUTPUT_FILE = "results.csv"