# 프로젝트의 모든 설정값 관리

# --- 1. 모델 및 API 설정 ---
# 사용할 GPT 모델 (e.g., "gpt-4-turbo", "gpt-5-nano", "gpt-3.5-turbo")
MODEL_NAME = "gpt-5-nano" 

# API 호출 간 대기 시간 (초). Rate Limit(요청 한도)에 맞춰 조절
SLEEP_TIME = 0.5

# --- 2. 파일 이름 설정 ---
# 'predict.py'가 읽을 원본 데이터 파일
INPUT_FILE = "dataset.csv" 

# 'predict.py'가 생성할 예측 결과 파일
OUTPUT_FILE = "results.csv"