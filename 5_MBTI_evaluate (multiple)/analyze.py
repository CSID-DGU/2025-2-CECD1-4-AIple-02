import pandas as pd
import config

def calculate_custom_metrics(df, score_col, total_trial_col):
    total_cnt = 0       # 전체 데이터 수
    drop_cnt = 0        # 동점(판단 보류) 수
    valid_cnt = 0       # 유효 데이터 수 (전체 - 동점)
    
    success_cnt = 0     # 성공(과반수 Output) 수
    perfect_cnt = 0     # 만장일치 성공 수

    for _, row in df.iterrows():
        score = row[score_col]
        trials = row[total_trial_col]
        
        if trials < config.TRIAL_COUNT: continue 
        
        total_cnt += 1
        ratio = score / trials

        # 1. 동점 체크 (continue로 인해 valid_cnt 증가 안 함)
        if ratio == 0.5:
            drop_cnt += 1
            continue
        
        # 유효 데이터 진입
        valid_cnt += 1

        # 2. 성공 체크 (과반수 이상)
        if ratio > 0.5:
            success_cnt += 1
            # 3. 만장일치 체크
            if ratio == 1.0:
                perfect_cnt += 1

    # 지표 계산 (0으로 나누기 방지)
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
    except Exception as e:
        print(f">> 오류: {e}")
        return

    print(f"\n=== MBTI 변환 성과 분석 리포트 ===")
    print(f"파일명: {config.OUTPUT_FILE}")
    print(f"시행 횟수(N): {config.TRIAL_COUNT}\n")

    dims = [("I vs E", "votes_output_ie"), ("N vs S", "votes_output_ns"), ("T vs F", "votes_output_tf")]

    for name, col in dims:
        if col not in df.columns: continue
        
        res = calculate_custom_metrics(df, col, 'valid_vote_count')
        
        print(f"■ [{name}] 분석 결과")
        
        # 지표 1: 유효 판정 비율
        print(f"  1. 유효 판정 비율: {res['decision_rate']*100:.1f}%")
        print(f"     - 계산: 유효 데이터({res['valid']}) / 전체 데이터({res['total']})")
        print(f"     - 제외됨(동점): {res['drop']}건")
        
        # 지표 2: 변환 성공률
        print(f"  2. 변환 성공률: {res['success_rate']*100:.1f}%")
        print(f"     - 계산: 성공({res['success']}) / 유효 데이터({res['valid']})")
        
        # 지표 3: 완벽 성공 비율
        print(f"  3. 완벽 성공 비율: {res['perfect_rate']*100:.1f}%")
        print(f"     - 계산: 만장일치({res['perfect']}) / 성공({res['success']})")
        
        print("-" * 50)

if __name__ == "__main__":
    analyze_metrics()