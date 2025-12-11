import pandas as pd
from sklearn.metrics import classification_report
import config

def analyze_metrics():
    try:
        df = pd.read_csv(config.OUTPUT_FILE)
    except Exception as e:
        print(f">> 오류: 결과 파일({config.OUTPUT_FILE})을 읽을 수 없습니다. {e}")
        return

    # 평가할 3가지 차원 매핑 (target_XX vs pred_XX)
    dimensions = [
        ("I vs E", "target_ie", "pred_ie"), 
        ("N vs S", "target_ns", "pred_ns"),
        ("T vs F", "target_tf", "pred_tf")
    ]

    print(f"\n=== 📊 MBTI 변환 모듈 성능 평가 리포트 ===")
    print(f"파일명: {config.OUTPUT_FILE}\n")

    for dim_name, target_col, pred_col in dimensions:
        print(f"----------------------------------------")
        print(f"   [{dim_name}] 평가")
        print(f"----------------------------------------")
        
        # 컬럼 존재 확인
        if target_col not in df.columns or pred_col not in df.columns:
            print(f">> [Skip] '{target_col}' 또는 '{pred_col}' 컬럼이 없어 건너뜁니다.")
            continue
            
        # 결측치(None/NaN) 제거
        temp_df = df.dropna(subset=[target_col, pred_col])
        
        y_true = temp_df[target_col]
        y_pred = temp_df[pred_col]
        
        if len(y_true) == 0:
            print(">> 유효한 데이터가 없습니다.")
            continue

        # 리포트 출력
        labels = sorted(list(set(y_true) | set(y_pred)))
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
        print("\n")

if __name__ == "__main__":
    analyze_metrics()