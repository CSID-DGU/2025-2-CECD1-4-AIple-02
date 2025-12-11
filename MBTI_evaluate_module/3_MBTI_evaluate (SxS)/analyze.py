import pandas as pd
import config

def calculate_success_rate(df, dimension_name, winner_col):
    """
    성공률(Success Rate) 계산: (Output 선택 횟수) / (전체 유효 평가 수)
    """
    # 결과가 있는 행만 필터링
    valid_df = df.dropna(subset=[winner_col])
    total_count = len(valid_df)
    
    if total_count == 0:
        return 0.0, 0, 0

    # Output이 이긴 경우 = 변환 성공 (Target에 더 근접함)
    wins_output = len(valid_df[valid_df[winner_col] == 'Output'])
    # Input이 이긴 경우 = 변환 실패 (원본이 더 낫거나 변환이 이상함)
    wins_input = len(valid_df[valid_df[winner_col] == 'Input'])

    # 성공률 계산
    success_rate = wins_output / total_count
    
    return success_rate, wins_output, wins_input

def analyze_metrics():
    try:
        df = pd.read_csv(config.OUTPUT_FILE)
    except Exception as e:
        print(f">> 오류: 결과 파일({config.OUTPUT_FILE})을 읽을 수 없습니다. {e}")
        return

    print(f"\n=== 📊 SxS 평가 리포트 ===")
    print(f"총 샘플 수: {len(df)}")
    print(f"성공률 = Output이 Target에 더 가깝다고 평가된 비율\n")

    dimensions = [
        ("I vs E", "winner_ie"),
        ("N vs S", "winner_ns"),
        ("T vs F", "winner_tf")
    ]

    total_score_sum = 0
    
    for dim_name, col_name in dimensions:
        if col_name not in df.columns:
            continue

        rate, n_success, n_fail = calculate_success_rate(df, dim_name, col_name)
        total_score_sum += rate
        
        print(f"[{dim_name}] 성공률: {rate:.4f} ({rate*100:.1f}%)")
        print(f" - 상세: Output승({n_success}), Input승({n_fail})")
        print("-" * 30)

    avg_rate = total_score_sum / 3
    print(f"\n[🏆 종합 평균 성공률]: {avg_rate:.4f}")

if __name__ == "__main__":
    analyze_metrics()