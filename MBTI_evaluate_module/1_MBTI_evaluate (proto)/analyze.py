import pandas as pd
from sklearn.metrics import classification_report, f1_score

import config

def analyze_metrics():
    try:
        # config.py에서 결과 파일 이름 사용
        df = pd.read_csv(config.OUTPUT_FILE)
    except FileNotFoundError:
        print(f">> 오류: '{config.OUTPUT_FILE}' 파일을 찾을 수 없습니다.")
        print(">> 'predict.py'를 먼저 실행하여 결과 파일을 생성했는지 확인하세요.")
        return

    # --- 1. API/파싱 오류가 발생한 행 평가에서 제외 ---
    original_count = len(df)
    error_mask = df['predicted_label'].isin(['API_ERROR', 'PARSE_ERROR'])
    df_cleaned = df[~error_mask]
    error_count = original_count - len(df_cleaned)

    if error_count > 0:
        print(f">> 알림: 총 {original_count}개 중 {error_count}개의 API/파싱 오류가 발생하여 평가에서 제외합니다.")

    # --- 2. 빈 칸(NaN)이 있는 행 평가에서 제외 ---
    original_cleaned_count = len(df_cleaned)
    # 'true_label' 또는 'predicted_label' 열에 비어있는 값(NaN)이 있으면 그 행을 삭제
    df_cleaned = df_cleaned.dropna(subset=['true_label', 'predicted_label'])
    nan_count = original_cleaned_count - len(df_cleaned)

    if nan_count > 0:
        print(f">> 알림: {nan_count}개의 행에 'true_label' 또는 'predicted_label' 값이 비어있어 평가에서 제외합니다.")
    # --- ---
    
    if df_cleaned.empty:
        print(">> 오류: 평가할 수 있는 유효한 데이터가 없습니다.")
        return

    # 'true_label' 컬럼(정답)이 있는지 확인
    if 'true_label' not in df_cleaned.columns:
        print(f">> 오류: {config.OUTPUT_FILE} 파일에 'true_label' 컬럼이 없습니다.")
        print(f">> {config.INPUT_FILE}에 'true_label' 정답 컬럼이 있는지 확인하세요.")
        return

    y_true = df_cleaned['true_label']
    y_pred = df_cleaned['predicted_label']

    labels = sorted(list(set(y_true) | set(y_pred)))
    
    print(f"\n--- '심판 LLM' 성능 평가 보고서 (기준: {config.OUTPUT_FILE}) ---")
    print(f"(총 {len(df_cleaned)}개 유효 데이터 기준)\n")

    report = classification_report(
        y_true, 
        y_pred, 
        labels=labels, 
        target_names=labels,
        digits=4,
        zero_division=0
    )
    
    print(report)

    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print("---")
    print(f">> 전체 F1 Score (Weighted Avg): {f1_weighted:.4f}")
    print("-------------------------------------------------------")


if __name__ == "__main__":
    analyze_metrics()