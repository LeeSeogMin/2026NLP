# 6장 리서치: LLM API 활용과 프롬프트 엔지니어링

## 검증된 참고문헌

1. Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS 2020*. https://arxiv.org/abs/2005.14165
2. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*. https://arxiv.org/abs/2201.11903
3. Kojima, T., et al. (2022). Large Language Models are Zero-Shot Reasoners. *NeurIPS 2022*. https://arxiv.org/abs/2205.11916
4. Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *NeurIPS 2023*. https://arxiv.org/abs/2305.10601
5. Sahoo, P., et al. (2024). A Systematic Survey of Prompt Engineering in Large Language Models. https://arxiv.org/abs/2402.07927

## API 가격 비교 (2026년 초 기준)

| 모델 | 입력 (1M 토큰) | 출력 (1M 토큰) | 컨텍스트 |
|------|--------------|--------------|---------|
| GPT-4o | $2.50 | $10.00 | 128K |
| GPT-4o-mini | $0.15 | $0.60 | 128K |
| Claude Sonnet 4.5 | $3.00 | $15.00 | 200K |
| Claude Haiku 4.5 | $1.00 | $5.00 | 200K |

## SDK 핵심 패턴 요약

### OpenAI vs Anthropic 주요 차이

| 항목 | OpenAI | Anthropic |
|------|--------|-----------|
| System 메시지 | messages 배열 내 role:"system" | 별도 system= 파라미터 |
| Tool 결과 역할 | role:"tool" | role:"user" + tool_result 블록 |
| 종료 사유 | finish_reason == "tool_calls" | stop_reason == "tool_use" |
| Tool 인자 | JSON 문자열 (파싱 필요) | 딕셔너리 (즉시 사용) |
| 토큰 사전 계산 | 클라이언트(tiktoken) | 서버(count_tokens API) |
| Structured Output | response_format=Model | output_format=Model |
| max_tokens | 선택 (기본값 있음) | 필수 |
