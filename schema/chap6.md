# 6장 집필계획서: LLM API 활용과 프롬프트 엔지니어링

## 기본 정보
- **주차**: 6주차
- **주제**: LLM API 활용과 프롬프트 엔지니어링
- **미션**: 수업이 끝나면 LLM API로 나만의 AI 앱 프로토타입을 만든다
- **목표 분량**: 600~700줄 (핵심 기술 장)
- **참고 자산**: 구 ch13 일부 (프롬프트 엔지니어링) + 신규 내용 대거 추가 (API, Function Calling, Structured Output)
- **GPU 필요도**: CPU 충분 (API 호출 기반)

## 학습 목표 (5개)
1. 주요 LLM API(OpenAI, Anthropic)의 호출 방식과 토큰 과금 구조를 이해한다
2. Zero-shot, Few-shot, Chain-of-Thought 프롬프팅 기법의 차이를 비교하고 적용할 수 있다
3. Structured Output(JSON Mode, Pydantic)으로 LLM 출력을 구조화할 수 있다
4. Function Calling을 구현하여 LLM에 외부 도구를 연동할 수 있다
5. LLM-as-a-Judge 패턴으로 모델 출력을 자동 평가할 수 있다

## 교시 구성

### 1교시: LLM API와 프롬프트 엔지니어링 (00:00~00:50)
- 6.1 상용 LLM API 생태계
  - 직관적 이해: 택시 vs 엔진 직접 제작
  - 표 6.1 API 제공자 비교 (OpenAI/Anthropic/Google/오픈소스)
  - API 호출 구조 (SDK 패턴, 응답 구조)
  - API Key 관리 (.env + python-dotenv)
  - 토큰 과금, Rate Limiting
  - Mermaid: fig-6-1-llm-api-ecosystem
- 6.2 프롬프트 엔지니어링
  - 직관적 이해: "AI에게 일 잘 시키는 기술"
  - System Prompt + Role Prompting
  - Zero-shot / Few-shot / Chain-of-Thought / Self-Consistency
  - 표 6.2 프롬프팅 기법 비교
  - CoT 전후 비교 실행 결과
  - Mermaid: fig-6-2-prompt-techniques

### 2교시: Structured Output과 LLM 평가 (01:00~01:50)
- 6.3 Structured Output과 Function Calling
  - 직관적 이해: "에세이 작가 vs 양식 작성자" + "LLM에게 팔다리 달아주기"
  - JSON Mode / Structured Output (Pydantic)
  - Function Calling 4단계 흐름 (날씨 조회 예제)
  - 출력 파싱 및 검증
  - 라이브 코딩 시연: Function Calling 날씨 조회
  - Mermaid: fig-6-3-function-calling-flow (시퀀스 다이어그램)
- 6.4 LLM 평가 기초
  - Perplexity, BLEU, ROUGE
  - LLM-as-a-Judge 패턴 + 실행 결과
  - Hallucination 탐지
  - Mermaid: fig-6-4-evaluation-pipeline

### 3교시: API 활용 실습 (02:00~02:50)
- 6.5 실습
  - Copilot 활용 안내
  - 프롬프팅 기법 비교 실험
  - Function Calling + Structured Output 실습
  - 과제: 도메인 특화 텍스트 분석 시스템

## 실습 코드
| 파일 | 내용 |
|------|------|
| 6-1-api기초.py | OpenAI/Claude API 기본 호출, temperature/max_tokens 파라미터, 모델 비교, 토큰 비용 계산 |
| 6-3-function-calling.py | JSON Mode, Structured Output (Pydantic), Function Calling (날씨 조회), 파싱/검증 |
| 6-5-프롬프트실습.py | 프롬프팅 기법 비교(Zero/Few/CoT), System Prompt 실험, LLM-as-Judge, 종합 미니앱 |

## 그래픽
| 파일 | 유형 | 내용 |
|------|------|------|
| fig-6-1-llm-api-ecosystem.mmd | flowchart LR | API 호출 흐름 (앱→SDK→클라우드→응답) |
| fig-6-2-prompt-techniques.mmd | flowchart TB | Zero-shot→Few-shot→CoT 계층 비교 |
| fig-6-3-function-calling-flow.mmd | sequenceDiagram | FC 4단계 (사용자→앱→LLM→도구→LLM→사용자) |
| fig-6-4-evaluation-pipeline.mmd | flowchart LR | 자동 평가 + LLM-as-Judge + 사람 평가 |
