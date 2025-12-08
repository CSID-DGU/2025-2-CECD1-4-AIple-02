import os
import openai

#파일명 함수명 유의!
from MBTI_dataset_preprocessing import run_preprocessing
from MBTI_dataset_expansion import run_expansion
from MBTI_dataset_convert import run_neutralization

def main():
    #아래 경로와 API키 입력
    
    BASE_DIR = r""
    
    MY_API_KEY = "" 

    FILE_RAW      = os.path.join(BASE_DIR, "mbtibench.jsonl")       #0.원본 데이터
    FILE_CLEAN    = os.path.join(BASE_DIR, "mbti_clean.jsonl")      #1.전처리 완료
    FILE_EXPANDED = os.path.join(BASE_DIR, "mbti_expanded.jsonl")   #2.문장 추출 완료
    FILE_FINAL    = os.path.join(BASE_DIR, "mbti_neutralized.csv")  #3.최종 결과물

    print("\n" + "="*50)
    print("MBTI 데이터셋 자동화 파이프라인 시작")
    print("="*50 + "\n")

    #1단계: 데이터 전처리
    if os.path.exists(FILE_RAW):
        run_preprocessing(FILE_RAW, FILE_CLEAN)
    else:
        print(f"[Error] 원본 파일이 없습니다: {FILE_RAW}")
        return

    #2단계: mbti가 잘 드러나는 문장 추출

    #전처리가 잘 됐는지 확인 후 진행
    if os.path.exists(FILE_CLEAN):
        print("\n--- [Step 2 진입] ---")
        run_expansion(FILE_CLEAN, FILE_EXPANDED, MY_API_KEY)
    else:
        print("[Error] Step 1 결과물이 없어 Step 2를 진행할 수 없습니다.")
        return

    #2단계: 밋밋한 문장, mbti가 잘 드러나는 문장 쌍 구성
    
    # 문장 추출이 잘 됐는지 확인 후 진행
    if os.path.exists(FILE_EXPANDED):
        print("\n--- [Step 3 진입] ---")
        run_neutralization(FILE_EXPANDED, FILE_FINAL, MY_API_KEY)
    else:
        print("[Error] Step 2 결과물이 없어 Step 3를 진행할 수 없습니다.")
        return

    print("\n" + "="*50)
    print(f"모든 작업 완료! 최종 파일: {FILE_FINAL}")
    print("="*50)

if __name__ == "__main__":
    main()