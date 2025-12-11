import pandas as pd
import config

def calculate_custom_metrics(df, vote_col, total_col):
    """
    3대 지표 계산 (동점 제외)
    """
    total_cnt = 0       
    drop_cnt = 0        
    valid_cnt = 0       
    success_cnt = 0     
    perfect_cnt = 0     

    for _, row in df.iterrows():
        v = row[vote_col]
        t = row[total_col]
        
        if t < config.TRIAL_COUNT: continue 
        
        total_cnt += 1
        ratio = v / t

        if ratio == 0.5:
            drop_cnt += 1
            continue
        
        valid_cnt += 1

        if ratio > 0.5:
            success_cnt += 1
            if ratio == 1.0:
                perfect_cnt += 1

    decision_rate = valid_cnt / total_cnt if total_cnt > 0 else 0
    success_rate = success_cnt / valid_cnt if valid_cnt > 0 else 0
    perfect_rate = perfect_cnt / success_cnt if success_cnt > 0 else 0

    return {
        "total": total_cnt,
        "valid": valid_cnt,
        "drop": drop_cnt,
        "success": success_cnt,
        "perfect": perfect_cnt,
        "decision_rate": decision_rate,
        "success_rate": success_rate,
        "perfect_rate": perfect_rate
    }

def analyze_metrics():
    try:
        df = pd.read_csv(config.OUTPUT_FILE)
        # 타겟 확인을 위해 원본(평가대상) 파일 로드
        df_input = pd.read_csv(config.INPUT_FILE)
    except Exception as e:
        print(f">> 오류: {e}")
        return

    print(f"\n=== 📊 MBTI Few-Shot 변환 성과 분석 리포트 ===")
    print(f"파일명: {config.OUTPUT_FILE}")
    print(f"시행 횟수(N): {config.TRIAL_COUNT}\n")

    dims = [
        ("I vs E", "votes_output_ie", "target_ie"), 
        ("N vs S", "votes_output_ns", "target_ns"), 
        ("T vs F", "votes_output_tf", "target_tf")
    ]

    for name, res_col, target_col in dims:
        if res_col not in df.columns: continue
        
        # 타겟이 있는 행만 필터링
        target_indices = df_input[df_input[target_col].notna()].index
        target_df = df.loc[target_indices]
        
        if len(target_df) == 0:
            print(f"■ [{name}] 데이터 없음 (Skip)")
            continue

        res = calculate_custom_metrics(target_df, res_col, 'valid_trial_count')
        
        print(f"■ [{name}] 분석 결과")
        print(f"  1. 유효 판정 비율: {res['decision_rate']*100:.1f}% ({res['valid']}/{res['total']})")
        print(f"     - 설명: 동점(1:1)을 제외한 데이터 수 / 전체 데이터 수")
        print(f"  2. 변환 성공률: {res['success_rate']*100:.1f}% ({res['success']}/{res['valid']})")
        print(f"     - 설명: 성공(2:0)한 데이터 수 / 유효 데이터 수")
        print(f"  3. 완벽 성공 비율: {res['perfect_rate']*100:.1f}% ({res['perfect']}/{res['success']})")
        print(f"     - 설명: 만장일치 데이터 수 / 성공한 데이터 수")
        print("-" * 50)

if __name__ == "__main__":
    analyze_metrics()