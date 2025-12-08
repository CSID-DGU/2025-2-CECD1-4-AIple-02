import pandas as pd
import re
import os

def hard_to_mbti(hard):
    #hard는 예: {'E/I':'I','S/N':'S','T/F':'T','J/P':'J'}
    #순서: E/I, S/N, T/F, J/P → 각각에서 선택된 글자만 이어붙임
    axis_order = [('E','I'), ('S','N'), ('T','F'), ('J','P')]
    letters = []
    #데이터가 딕셔너리인지 확인 (가끔 문자열로 들어오는 경우 대비)
    if not isinstance(hard, dict):
        return None
        
    for left, right in axis_order:
        key = f'{left}/{right}'
        if key in hard:
            letters.append(hard[key])
    return ''.join(letters)

def posts_to_text(x):
    if isinstance(x, list):
        return " ".join(map(str, x))
    s = str(x).strip()
    #양끝 [ ] 제거
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    return s

def remove_emoticons(text):
    #:happy:와 같은 이모티콘 변환된 단어 삭제
    text = re.sub(r':[a-zA-Z_]+:', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

#전처리 함수
def run_preprocessing(input_path, output_path):
    print(f"[Step 1] 전처리 시작: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"오류: 파일을 찾을 수 없습니다 -> {input_path}")
        return

    try:
        df = pd.read_json(input_path, lines=True, encoding="utf-8")
    except ValueError as e:
        print(f"JSON 읽기 오류: {e}")
        return

    #전처리 로직 적용
    df["HardMBTI"] = df["hardlabels"].apply(hard_to_mbti)
    df["posts"] = df["posts"].apply(posts_to_text)
    df["posts"] = df["posts"].apply(remove_emoticons)

    #데이터 정리
    out = df[["posts", "HardMBTI"]].dropna(subset=["posts", "HardMBTI"]).copy()
    out = out.rename(columns={"posts": "text", "HardMBTI": "label"})

    #저장
    out.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"[Step 1] 전처리 완료. 저장됨 -> {output_path}")

#단독 테스트용 메인함수
if __name__ == "__main__":
    #테스트용 경로
    test_in = r""
    test_out = r""
    run_preprocessing(test_in, test_out)