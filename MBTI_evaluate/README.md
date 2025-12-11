
predict.py 실행 시 result.csv 생성
analyze.py 실행 시 result.csv 분석

1. Proto 
predict: 문장 → 예상 MBTI 
analyze: F1 Score 도출
I/E 판별

2. F1 Score 
predict: 문장 → 예상 MBTI 
analyze: F1 Score 도출
I/E, N/S, T/F 판별

3. Side by Side
predict: 문장A, 문장B, target MBTI → target MBTI에 가까운 문장 택1 
analyze: 성공률 합산

4. Side by Side (2nd)
predict: 문장A, 문장B, target MBTI → target MBTI에 가까운 문장 택1 
analyze: 성공률 합산
Side by Side에서 프롬프트 수정

5. Multiple
predict: 문장A, 문장B, target MBTI → target MBTI에 가까운 문장 택1 
analyze: 성공률 합산
Side by Side에서 노이즈 제거 위해 반복 시행 기능 추가

6. Enhanced
predict: 문장A, 문장B, target MBTI → target MBTI에 가까운 문장 택1 
analyze: 성공률 합산
multiple에서 프롬프트 강화 (Zero-Shot 학습)

7. Few-Shot
predict: 문장A, 문장B, target MBTI, ground_truth → ground_truth에 가까운 문장 택1 
analyze: 성공률 합산
ground_truth에서 랜덤으로 n개의 예시 문장을 추출해 프롬프트로 전송
(ground_truth는 말투 변환 모듈에서 Few-Shot 예시 문장으로 사용하던 데이터셋)

8. Few-Shot (Chain of Thought)
predict: 문장A, 문장B, target MBTI, ground_truth → ground_truth에 가까운 문장 택1, 그 이유 반환
analyze: 성공률 합산
Few-Shot 모델에서 이유를 반환하는 기능 추가

9. Persona
predict: 문장열A, 문장열B, target MBTI, ground_truth → ground_truth에 가까운 문장열 택1 
analyze: 성공률 합산
이전까지와 달리 발화한 Agent 단위로 MBTI를 평가함
문장 단위로 평가 할 때에 비해 정확도 비약적 상승
