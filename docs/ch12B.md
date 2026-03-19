## 12주차 B회차: Agent 프로토타입 개발

> **미션**: LangGraph로 웹 검색·계산·DB 조회 도구를 활용하는 AI Agent를 구현하고 ReAct 루프를 분석할 수 있다

### 수업 타임라인

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 조 편성 + 역할 배분 (조원 A/B) | 사용 안 함 |
| 00:05~00:10 | A회차 핵심 리캡 + 과제 스펙 재확인 | 사용 안 함 |
| 00:10~00:55 | 2인1조 Copilot 구현 (체크포인트 3회) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (조별 1부) | 사용 안 함 |
| 01:00~01:20 | 결과 토론 (구현 전략 비교·성능 차이 분석) | 사용 안 함 |
| 01:20~01:28 | 핵심 정리 | 사용 안 함 |
| 01:28~01:30 | 다음 주 예고 | 사용 안 함 |

---

### A회차 핵심 리캡

**AI Agent의 정의와 특징**:
- Chatbot은 사용자 질문에 학습된 지식으로 바로 답변한다
- AI Agent는 목표를 분해하고, 필요한 도구(함수, API)를 자동으로 선택하여 호출하고, 결과를 통합한다
- Agent의 핵심 특징: 자율성, 목표 지향성, 도구 활용 능력, 반응성

**ReAct (Reasoning + Acting) 루프**:
- 네 단계의 반복: THOUGHT (생각) → ACTION (행동) → OBSERVATION (관찰) → (반복 또는 종료)
- 예: "2024년 한국 평균 기온은 몇 도였고, 2023년보다 몇 도 올랐나?"
  - THOUGHT: "두 연도의 기온 정보가 필요하다"
  - ACTION: web_search 도구 호출
  - OBSERVATION: 도구 결과 수집
  - (반복) THOUGHT: "계산이 필요하다" → ACTION: calculator 호출
  - ANSWER: 최종 답변 생성

**Tool(도구)의 역할**:
- Agent의 "팔과 눈" 역할을 하는 함수/API
- 웹 검색, 계산, 데이터베이스 조회, 파이썬 실행 등 다양한 도구 가능
- 좋은 Tool은 명확한 이름, 상세한 설명, 명확한 파라미터를 갖춘다

**LangGraph 기본 개념**:
- State: Agent가 유지하는 정보 (메시지, 도구 결과 등)
- Node: 각 상태에서 실행되는 함수 (LLM 실행, Tool 실행)
- Edge: Node 간의 전환 조건 (도구 필요 여부에 따라 다음 단계 결정)
- StateGraph: 이들을 조합한 전체 그래프

---

### 과제 스펙

**과제**: 도메인 특화 AI Agent 프로토타입 개발 + ReAct 루프 분석

**제출 형태**: 조별 1부, Google Classroom 업로드

**파일 구성**:
- 구현 코드 파일 (`*.py`)
- Agent 실행 로그 (각 질문별 THOUGHT/ACTION/OBSERVATION 기록)
- 분석 리포트 (1-2페이지)

**검증 기준**:
- ✓ 2~3개의 Tool 함수 정의 및 동작 확인
- ✓ AgentState, 조건부 라우팅(should_continue), Node 함수 구현
- ✓ StateGraph 컴파일 및 Agent 실행
- ✓ 3개 이상의 복잡한 질문에 대한 ReAct 루프 로그 기록
- ✓ Tool 호출 순서와 오류 처리 과정 분석

---

### 2인1조 실습

> **Copilot 활용**: 먼저 간단한 도구 함수를 정의한 뒤, Copilot에게 "이 도구들을 LangGraph에 연결해줄래?", "Agent가 질문을 받으면 어떤 도구를 부를지 판단하는 로직 추가해줄래?" 같이 단계적으로 요청한다. Copilot의 제안을 검토하면서 Agent의 동작 원리를 깊이 있게 이해할 수 있다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 실행, Tool 정의 및 테스트
- **조원 B (네비게이터)**: 로직 검토, LangGraph 그래프 설계, 질문 시나리오 작성
- **체크포인트마다 역할 교대**: 드라이버와 네비게이터를 번갈아가며 진행하여 두 명 모두 전체 구현을 이해한다

---

#### 체크포인트 1: Tool 정의 + Agent State 구성 (15분)

**목표**: 3개의 도구 함수를 정의하고, LangGraph의 AgentState를 설계한다

**핵심 단계**:

① **Tool 함수 정의** — 각 도구는 명확한 설명과 함께 정의된다

```python
import json
from typing import Any, Dict

# Tool 1: 웹 검색 (시뮬레이션)
def web_search(query: str) -> str:
    """
    인터넷에서 주어진 쿼리를 검색한다.

    Args:
        query (str): 검색 키워드. 예: "2024년 한국 평균 기온"

    Returns:
        str: 검색 결과 요약

    예시:
        >>> web_search("2024년 한국 평균 기온")
        "기상청 발표: 2024년 한국 평균 기온은 14.8°C"
    """
    search_database = {
        "2024년 한국 평균 기온": "기상청 발표: 2024년 한국 평균 기온은 14.8°C (최근 30년 평균 12.8°C 대비 +2.0°C)",
        "2023년 한국 평균 기온": "기상청 발표: 2023년 한국 평균 기온은 13.5°C (최근 30년 평균 대비 +0.7°C)",
        "한국 GDP": "한국의 2024년 명목 GDP는 약 1.45 trillion 달러로 세계 12위",
        "일본 GDP": "일본의 2024년 명목 GDP는 약 4.23 trillion 달러로 세계 3위",
        "애플 주가": "현재 애플 주가: $145 (전일 대비 +2.3%)",
        "삼성전자 주가": "현재 삼성전자 주가: 62,500원 (전일 대비 -1.2%)",
    }

    for key, value in search_database.items():
        if key.lower() in query.lower() or query.lower() in key.lower():
            return value

    return f"'{query}'에 대한 검색 결과를 찾을 수 없습니다. 다른 검색어를 시도해주세요."


# Tool 2: 계산기
def calculator(expression: str) -> float:
    """
    산술 식을 계산한다.

    Args:
        expression (str): 계산할 식. 예: "14.8 - 13.5"

    Returns:
        float: 계산 결과

    예시:
        >>> calculator("14.8 - 13.5")
        1.3
    """
    try:
        # 안전한 계산을 위해 숫자와 연산자만 허용
        allowed_chars = set("0123456789.+-*/()")
        if not all(c in allowed_chars for c in expression):
            return f"오류: 허용되지 않는 문자가 포함됨"

        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"계산 오류: {e}"


# Tool 3: 데이터베이스 조회 (판매 데이터)
def query_sales_db(year: int, region: str) -> Dict[str, int]:
    """
    회사 판매 데이터베이스를 조회한다.

    Args:
        year (int): 조회 연도. 예: 2024
        region (str): 지역. 예: "한국", "일본", "미국"

    Returns:
        dict: {제품명: 매출액} 형식의 딕셔너리

    예시:
        >>> query_sales_db(2024, "한국")
        {"스마트폰": 1200000000, "태블릿": 350000000, "노트북": 450000000}
    """
    sales_database = {
        (2024, "한국"): {
            "스마트폰": 1200000000,
            "태블릿": 350000000,
            "노트북": 450000000,
            "이어폰": 200000000
        },
        (2023, "한국"): {
            "스마트폰": 1000000000,
            "태블릿": 300000000,
            "노트북": 400000000,
            "이어폰": 150000000
        },
        (2024, "일본"): {
            "스마트폰": 850000000,
            "태블릿": 280000000,
            "노트북": 320000000,
            "이어폰": 120000000
        },
        (2023, "일본"): {
            "스마트폰": 750000000,
            "태블릿": 250000000,
            "노트북": 280000000,
            "이어폰": 100000000
        },
    }

    key = (year, region)
    if key in sales_database:
        return sales_database[key]
    else:
        return {"오류": f"{year}년 {region} 판매 데이터를 찾을 수 없습니다"}
```

② **AgentState 정의** — Agent가 유지할 모든 정보를 구조화한다

```python
from typing import TypedDict, Annotated, Optional, List

class AgentState(TypedDict):
    """Agent의 상태를 정의한다"""

    # 입력/출력
    input: str  # 사용자의 질문
    output: Optional[str]  # 최종 답변

    # 대화 기록
    messages: Annotated[List[dict], "LLM과의 주고받음"]

    # 도구 실행 결과
    tool_results: Annotated[Dict[str, Any], "각 도구의 실행 결과"]

    # 반복 제어
    iterations: int  # 지금까지 몇 번 루프를 돌았는가
    max_iterations: int  # 최대 허용 반복 횟수
```

**검증 체크리스트**:
- [ ] web_search, calculator, query_sales_db 세 도구가 모두 정의되었는가?
- [ ] 각 도구의 docstring이 명확한가? (목적, 파라미터, 반환값, 예시)
- [ ] 각 도구를 단독으로 실행하여 동작을 확인했는가?
  - `web_search("한국 GDP")` → 결과 출력
  - `calculator("1200000000 - 1000000000")` → 결과 출력
  - `query_sales_db(2024, "한국")` → 결과 출력
- [ ] AgentState가 input, messages, tool_results, iterations를 모두 포함하는가?

**Copilot 프롬프트 1**:
```
"다음 세 가지 도구를 Python 함수로 구현해줄래?
1. web_search(query: str) - 검색어를 받아 관련 정보 반환
2. calculator(expression: str) - 산술식을 받아 계산 결과 반환
3. query_sales_db(year: int, region: str) - 연도와 지역을 받아 판매 데이터 반환

각 도구는 mock 데이터를 포함해야 하고, 명확한 docstring을 가져야 해."
```

**Copilot 프롬프트 2**:
```
"위의 도구들을 사용하는 AI Agent를 위해 TypedDict로 AgentState를 정의해줄래?
State는 사용자 입력, 대화 메시지 목록, 도구 결과, 반복 횟수를 포함해야 해."
```

---

#### 체크포인트 2: ReAct 루프 구현 + LangGraph 구성 (15분)

**목표**: Agent의 LLM 실행 Node, Tool 실행 Node, 조건부 라우팅 함수를 구현하고, StateGraph를 컴파일한다

**핵심 단계**:

① **조건부 라우팅 함수** — Agent가 다음 단계를 결정한다

```python
def should_continue(state: AgentState) -> str:
    """
    현재 상태를 보고 다음 Node를 결정한다.

    Returns:
        str: "llm" (LLM 실행), "tools" (Tool 호출), "end" (종료)
    """
    # 최대 반복 횟수 도달 시 종료
    if state["iterations"] >= state["max_iterations"]:
        print(f"[INFO] 최대 반복 횟수({state['max_iterations']}) 도달. Agent 종료.")
        return "end"

    # 최근 LLM 응답 확인
    if not state["messages"]:
        return "llm"

    latest_message = state["messages"][-1].get("content", "")

    # "최종 답변:" 또는 "ANSWER:"로 시작하면 종료
    if "최종 답변:" in latest_message or "ANSWER:" in latest_message:
        state["output"] = latest_message.replace("최종 답변:", "").replace("ANSWER:", "").strip()
        print(f"[INFO] 최종 답변 생성. Agent 종료.")
        return "end"

    # "<tool_call>"을 포함하면 Tool 실행
    if "<tool_call>" in latest_message:
        print(f"[INFO] Tool 호출 감지. Tool 실행 Node로 이동.")
        return "tools"

    # 그 외에는 LLM을 다시 실행
    return "llm"
```

② **LLM 실행 Node** — Agent가 생각하고 행동을 결정한다

```python
def run_llm_node(state: AgentState) -> AgentState:
    """
    LLM을 실행하여 다음 행동을 결정한다.
    """
    print(f"\n=== 반복 {state['iterations'] + 1} ===")
    print(f"[LLM Node] LLM에 현재까지의 메시지 전달")

    # 여기서는 LLM 대신 간단한 규칙 기반 로직을 사용
    # 실제로는 ChatOpenAI나 다른 LLM 모델을 사용

    current_input = state["messages"][-1].get("content", "")

    # Tool 결과가 있는 경우
    if state["tool_results"]:
        # LLM이 도구 결과를 보고 판단한다
        tool_results_text = "\n".join([
            f"{tool}: {result}"
            for tool, result in state["tool_results"].items()
        ])

        thought = f"""[THOUGHT]
도구 실행 결과를 받았습니다:
{tool_results_text}

이 결과를 분석하여 최종 답변을 생성하겠습니다.
"""

        # 최종 답변 생성
        final_answer = f"""[ANSWER]
사용자의 질문에 대한 답변:
{state['input']}

도구 결과 분석:
{tool_results_text}

최종 답변: [분석 과정을 거친 최종 답변]
"""

        state["messages"].append({
            "role": "assistant",
            "content": final_answer
        })

    else:
        # 처음 시작이거나 중간 단계
        thought = f"""[THOUGHT]
사용자의 질문: "{current_input}"

이 질문을 답하려면 어떤 정보가 필요할까?
1. 먼저 웹에서 관련 정보를 검색해보자.
2. 필요하면 계산이나 데이터 조회를 추가로 진행한다.
"""

        # Tool 호출 결정
        if "기온" in current_input:
            tool_call = """[ACTION]
<tool_call>{"tool": "web_search", "input": {"query": "2024년 한국 평균 기온"}}</tool_call>
<tool_call>{"tool": "web_search", "input": {"query": "2023년 한국 평균 기온"}}</tool_call>
"""

        elif "GDP" in current_input:
            tool_call = """[ACTION]
<tool_call>{"tool": "web_search", "input": {"query": "한국 GDP"}}</tool_call>
<tool_call>{"tool": "web_search", "input": {"query": "일본 GDP"}}</tool_call>
"""

        elif "매출" in current_input or "판매" in current_input:
            tool_call = """[ACTION]
<tool_call>{"tool": "query_sales_db", "input": {"year": 2024, "region": "한국"}}</tool_call>
<tool_call>{"tool": "query_sales_db", "input": {"year": 2023, "region": "한국"}}</tool_call>
"""

        else:
            tool_call = """[ACTION]
<tool_call>{"tool": "web_search", "input": {"query": "일반적인 정보"}}</tool_call>
"""

        state["messages"].append({
            "role": "assistant",
            "content": thought + "\n" + tool_call
        })

    state["iterations"] += 1
    return state


def execute_tools_node(state: AgentState) -> AgentState:
    """
    LLM이 요청한 도구들을 실행한다.
    """
    print(f"[Tools Node] Tool 실행 시작")

    # 최근 LLM 응답에서 Tool 호출 추출
    latest_response = state["messages"][-1].get("content", "")

    # <tool_call>...</tool_call> 파싱
    import re
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', latest_response, re.DOTALL)

    tool_results = {}

    for match in matches:
        try:
            tool_json = json.loads(match)
            tool_name = tool_json.get("tool", "")
            tool_input = tool_json.get("input", {})

            print(f"  Tool: {tool_name}, Input: {tool_input}")

            # 각 도구 실행
            if tool_name == "web_search":
                result = web_search(tool_input.get("query", ""))

            elif tool_name == "calculator":
                result = calculator(tool_input.get("expression", ""))

            elif tool_name == "query_sales_db":
                result = query_sales_db(
                    tool_input.get("year", 2024),
                    tool_input.get("region", "한국")
                )

            else:
                result = f"알 수 없는 도구: {tool_name}"

            tool_results[tool_name] = result
            print(f"    → 결과: {result}")

        except json.JSONDecodeError as e:
            print(f"  JSON 파싱 오류: {e}")
            tool_results["error"] = f"JSON 파싱 오류: {e}"

    # Agent state 업데이트
    state["tool_results"].update(tool_results)

    # 도구 결과를 메시지에 추가
    state["messages"].append({
        "role": "tool_results",
        "content": str(tool_results)
    })

    return state
```

③ **StateGraph 구성** — 전체 흐름을 정의한다

```python
from langgraph.graph import StateGraph, END

def build_agent_graph():
    """
    ReAct Agent의 StateGraph를 구성하고 컴파일한다.
    """
    # 그래프 생성
    graph = StateGraph(AgentState)

    # Node 추가
    graph.add_node("llm", run_llm_node)
    graph.add_node("tools", execute_tools_node)

    # 시작점 설정
    graph.set_entry_point("llm")

    # Edge 추가
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "llm": "llm",
            "end": END
        }
    )

    # Tool 실행 후 다시 LLM으로
    graph.add_edge("tools", "llm")

    # 컴파일
    agent = graph.compile()

    return agent
```

**검증 체크리스트**:
- [ ] should_continue 함수가 "llm", "tools", "end" 중 하나를 반환하는가?
- [ ] run_llm_node가 Agent state의 messages를 수정하고 반환하는가?
- [ ] execute_tools_node가 <tool_call> 형식의 Tool 호출을 파싱하는가?
- [ ] execute_tools_node가 각 도구를 실행하고 결과를 state["tool_results"]에 저장하는가?
- [ ] StateGraph가 컴파일되는가?

**Copilot 프롬프트 3**:
```
"LangGraph를 사용하여 ReAct Agent를 구현해줄래?
- should_continue 함수로 다음 step 결정
- run_llm_node로 LLM이 <tool_call>...</tool_call> 형식으로 tool 호출 요청
- execute_tools_node로 실제 tool 실행
- StateGraph로 llm → tools → llm 루프 구성"
```

**Copilot 프롬프트 4**:
```
"위의 Node들을 StateGraph에 연결해서 Agent를 완성해줄래?
Entry point는 'llm'이고, should_continue 함수로 조건부 라우팅을 해야 해."
```

---

#### 체크포인트 3: Agent 테스트 + ReAct 루프 로그 분석 (10분)

**목표**: 3개 이상의 복잡한 질문으로 Agent를 테스트하고, 각 질문의 THOUGHT-ACTION-OBSERVATION 과정을 기록하며 분석한다

**핵심 단계**:

① **Agent 초기화 및 실행**

```python
def initialize_agent_state(user_input: str) -> AgentState:
    """
    Agent의 초기 상태를 설정한다.
    """
    return {
        "input": user_input,
        "output": None,
        "messages": [{"role": "user", "content": user_input}],
        "tool_results": {},
        "iterations": 0,
        "max_iterations": 10
    }


def run_agent_with_logging(agent, user_input: str) -> Dict[str, Any]:
    """
    Agent를 실행하고 전체 과정을 로깅한다.
    """
    print(f"\n{'='*60}")
    print(f"질문: {user_input}")
    print(f"{'='*60}")

    initial_state = initialize_agent_state(user_input)

    try:
        result = agent.invoke(initial_state)

        print(f"\n{'─'*60}")
        print(f"[최종 결과]")
        print(f"반복 횟수: {result['iterations']}")
        print(f"최종 답변: {result.get('output', '답변 없음')}")
        print(f"{'─'*60}")

        return result

    except Exception as e:
        print(f"[오류] Agent 실행 중 예외 발생: {e}")
        return initial_state
```

② **테스트 질문 3가지**

```python
def test_agent():
    """
    Agent를 여러 질문으로 테스트한다.
    """
    # Agent 구성
    agent = build_agent_graph()

    # 테스트 질문
    test_queries = [
        "2024년 한국 평균 기온은 몇 도였고, 2023년보다 몇 도 올랐나?",
        "2024년 한국의 전체 매출은 얼마인가? (스마트폰+태블릿+노트북 합산)",
        "한국 GDP와 일본 GDP를 비교하면 어느 쪽이 더 큰가?"
    ]

    results = []

    for query in test_queries:
        result = run_agent_with_logging(agent, query)
        results.append(result)

    return results
```

③ **ReAct 루프 분석**

```python
def analyze_react_loop(result: AgentState):
    """
    각 질문의 ReAct 루프 과정을 분석한다.
    """
    print(f"\n[ReAct 루프 분석]")
    print(f"총 반복 횟수: {result['iterations']}")

    for i, msg in enumerate(result['messages']):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')[:100]  # 처음 100자만 출력
        print(f"  [{i}] {role}: {content}...")

    print(f"\n[도구 사용]")
    for tool, tool_result in result['tool_results'].items():
        tool_result_str = str(tool_result)[:80]
        print(f"  {tool}: {tool_result_str}...")
```

④ **실행 및 로그 기록**

```python
# Agent 테스트
print("="*60)
print("ReAct Agent 테스트 시작")
print("="*60)

results = test_agent()

# 각 결과 분석
for i, result in enumerate(results, 1):
    print(f"\n[질문 {i} 분석]")
    analyze_react_loop(result)
```

**검증 체크리스트**:
- [ ] Agent가 3개 질문 모두에 대해 실행되었는가?
- [ ] 각 질문에서 도구가 실제로 호출되었는가?
- [ ] 도구 결과가 agent state의 tool_results에 저장되었는가?
- [ ] 반복 횟수가 max_iterations을 초과하지 않았는가?

**Copilot 프롬프트 5**:
```
"위의 Agent로 다음 세 가지 질문을 테스트해줄래?
1. '2024년 한국 평균 기온은 몇 도였고, 2023년보다 몇 도 올랐나?'
2. '한국과 일본의 GDP를 비교해줄래?'
3. '2024년 한국의 전체 매출 (스마트폰+태블릿+노트북)은?'

각 질문별로 THOUGHT-ACTION-OBSERVATION 과정을 출력해줘."
```

**Copilot 프롬프트 6** (선택):
```
"Agent가 tool 호출 오류를 받았을 때 자동으로 재시도하는 로직을 추가해줄래?
예를 들어, query_sales_db가 실패하면 다른 방식으로 데이터를 조회하도록."
```

---

### 제출 안내 (Google Classroom)

**제출 방법**:
- Google Classroom의 "12주차 B회차" 과제에 조별 1부 제출
- 파일명 형식: `group_{조번호}_ch12B.zip`

**포함할 파일**:
```
group_{조번호}_ch12B/
├── ch12B_react_agent.py           # 전체 구현 코드
├── agent_test_log.txt             # Agent 실행 로그 (텍스트)
├── react_loop_analysis.md         # ReAct 루프 분석 문서
└── report.md                       # 분석 리포트 (1-2페이지)
```

**코드 파일 포함 항목** (`ch12B_react_agent.py`):
```
1. Tool 함수 정의
   - web_search(query: str) → str
   - calculator(expression: str) → float
   - query_sales_db(year: int, region: str) → dict

2. AgentState 정의
   - TypedDict로 정의

3. Node 함수
   - run_llm_node(state) → state
   - execute_tools_node(state) → state
   - should_continue(state) → str

4. StateGraph 구성
   - build_agent_graph() → compiled agent

5. 테스트 함수
   - test_agent()
   - run_agent_with_logging()
```

**로그 파일** (`agent_test_log.txt`):
- 3개 질문 각각에 대한 실행 로그
- 각 반복 단계별 THOUGHT, ACTION, OBSERVATION 기록
- 최종 답변 및 사용된 도구 목록

**리포트 포함 항목** (`report.md`):
- **1. Agent 설계**: 선택한 3개 도구와 각 도구의 용도 (3-4문장)
- **2. ReAct 루프 분석**: 질문 1개 선택 후, THOUGHT→ACTION→OBSERVATION 과정 상세 설명 (5-7문장)
- **3. Tool 호출 순서 분석**: "Agent가 왜 이 순서로 도구를 호출했는가?" (3-5문장)
- **4. Error Handling 사례**: Tool 호출 실패 시 Agent의 대응 (있다면 기록, 없다면 가상 시나리오 제시)
- **5. 성능 평가**: 최대 반복 횟수 대비 실제 반복 횟수, 효율성 분석 (2-3문장)
- **6. Copilot 활용 경험**: 어떤 프롬프트가 효과적이었는가? (2문장)

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 구현 결과를 공유하며, 다른 조의 Agent 설계와 Tool 선택을 비교하고, ReAct 루프의 효율성을 함께 분석한다

**토론 주제**:

① **Tool 선택의 다양성**
- 각 조가 선택한 3개 도구는 무엇인가?
- "왜 이 도구들을 선택했는가?"
- 다른 도구를 사용했다면 어떤 차이가 있었을까?

② **ReAct 루프의 효율성**
- 각 질문에서 평균 몇 번 반복하는가?
- 최소/최대 반복 횟수는?
- "왜 어떤 질문은 더 많이 반복하는가?"

③ **Tool Composition의 가치**
- 하나의 도구만 사용하는 경우 vs 여러 도구를 조합하는 경우
- 병렬 실행(동시 호출)과 순차 실행(차례로 호출)의 차이
- "만약 Tool 실행이 순서대로 아닌 병렬로 진행된다면?"

④ **Error Handling 전략**
- 도구 호출 실패 시 Agent의 대응 방식
- 재시도 로직의 필요성
- "어떤 오류는 자동으로 복구 가능하고, 어떤 오류는 사용자 개입이 필요한가?"

⑤ **LangGraph의 장점**
- 조건부 라우팅으로 얻은 이점
- Node와 Edge로 Agent 로직을 표현하는 이점
- "만약 LangGraph 없이 수동으로 루프를 구현했다면?"

**발표 형식**:
- 각 조 3~5분 발표 (Tool 선택 사유 + 주요 결과)
- 다른 조의 질문에 답변 (2~3개 질문)
- 교수의 보충 설명 및 피드백

---

### 흔한 오류 3가지 + 확인법

#### 오류 1: Tool 호출 JSON 파싱 실패

**오류 증상**:
```
JSON 파싱 오류: Expecting value: line 1 column 1 (char 0)
```

**원인**:
- LLM이 `<tool_call>...</tool_call>` 형식으로 정확히 응답하지 않음
- JSON이 불완전하거나 따옴표가 빠짐

**확인/해결법**:
```python
# 1단계: 응답 확인
latest_response = state["messages"][-1].get("content", "")
print(f"LLM 응답: {repr(latest_response)}")

# 2단계: <tool_call> 매칭 확인
import re
matches = re.findall(r'<tool_call>(.*?)</tool_call>', latest_response, re.DOTALL)
print(f"매칭된 문자열 개수: {len(matches)}")
for match in matches:
    print(f"  {match}")

# 3단계: JSON 직접 파싱 테스트
try:
    tool_json = json.loads(matches[0])
    print("JSON 파싱 성공")
except json.JSONDecodeError as e:
    print(f"JSON 파싱 실패: {e}")
```

#### 오류 2: Tool 함수 반환값 타입 불일치

**오류 증상**:
```
TypeError: unsupported operand type(s) for -: 'str' and 'float'
```

**원인**:
- Tool이 dict를 반환해야 하는데 str을 반환함
- Tool 결과를 state["tool_results"]에 저장할 때 타입이 맞지 않음

**확인/해결법**:
```python
# 1단계: Tool 반환값 확인
result = web_search("한국 GDP")
print(f"반환 타입: {type(result)}")
print(f"반환값: {result}")

# 2단계: Query_sales_db는 dict 반환
result = query_sales_db(2024, "한국")
print(f"반환 타입: {type(result)}")  # <class 'dict'>

# 3단계: Tool 결과 타입 확인
tool_results = state["tool_results"]
for tool, result in tool_results.items():
    print(f"{tool}: {type(result)} = {result}")
```

#### 오류 3: 무한 루프 (최대 반복 횟수 도달)

**오류 증상**:
```
[INFO] 최대 반복 횟수(10) 도달. Agent 종료.
```

**원인**:
- should_continue가 항상 "llm" 또는 "tools"를 반환
- "최종 답변:" 플래그를 찾지 못함

**확인/해결법**:
```python
# 1단계: should_continue 로직 확인
def should_continue(state: AgentState) -> str:
    latest = state["messages"][-1].get("content", "")
    print(f"[DEBUG] 최근 메시지: {latest[:100]}")

    if "최종 답변:" in latest:
        print("[DEBUG] 최종 답변 감지 → 'end'")
        return "end"
    elif "<tool_call>" in latest:
        print("[DEBUG] Tool 호출 감지 → 'tools'")
        return "tools"
    else:
        print("[DEBUG] 재LLM → 'llm'")
        return "llm"

# 2단계: 최대 반복 횟수 확인
if state["iterations"] >= state["max_iterations"]:
    print(f"[DEBUG] 반복 {state['iterations']} / {state['max_iterations']}")
    return "end"

# 3단계: Run_llm_node에서 "최종 답변:" 생성되는지 확인
print(f"[DEBUG] run_llm_node 내 생성 메시지:\n{state['messages'][-1].get('content', '')}")
```

---

### 다음 주 예고

다음 주 13주차 A회차에서는 **상황 인식 Agent (Situation Aware Agent)**를 배운다.

**예고 내용**:
- **더 많은 도구의 통합**: Tool이 5개 이상인 경우, Agent는 어떻게 올바른 도구를 선택하는가?
- **Memory와 Context**: Agent가 이전 대화 기록을 참고하여 현재 질문에 답하는 방식
- **Prompt Engineering for Agents**: Tool 설명을 더 정교하게 작성하여 Agent의 정확성을 높이는 방법
- **Multi-Agent**: 여러 Agent가 협력하여 복잡한 문제를 해결하는 구조
- PyTorch와 LangGraph를 결합한 고급 Agent 패턴

**사전 준비**:
- 12주차 B회차의 tool composition을 다시 검토
- 자신이 작성한 Agent의 문제점이나 개선할 점 정리해오기

---

## 참고 자료

**실습 코드**:
- _전체 구현 코드는 practice/chapter12/code/12-1-react-agent.py 참고_
- _고급 예제 (Error Handling)는 practice/chapter12/code/12-2-agent-with-retry.py 참고_
- _Tool Composition 예제는 practice/chapter12/code/12-3-multi-tool-agent.py 참고_

**권장 읽기**:
- Yao, S., Zhao, J., Yu, D., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *arXiv*. https://arxiv.org/abs/2210.03629
- LangChain. Tool Calling. https://python.langchain.com/docs/concepts/tool_calling/
- LangChain. LangGraph: Build Stateful, Multi-Actor Applications. https://python.langchain.com/docs/langgraph/
- OpenAI. Function Calling. https://platform.openai.com/docs/guides/function-calling
- Webson, A., & Pavlick, E. (2022). Do Prompt-Based Models Really Understand the Meaning of Their Prompts? *arXiv*. https://arxiv.org/abs/2109.01247

