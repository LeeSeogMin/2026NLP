# 제12장 C: ReAct AI Agent 구현 — 모범 구현과 해설

> 이 문서는 B회차 과제 제출 후 공개된다. 제출 전에는 열람하지 않는다.

---

## 체크포인트 1 모범 구현: Tool 정의 + Agent State 구성

ReAct Agent의 기초는 명확하고 신뢰할 수 있는 Tool 정의와 깔끔한 State 설계에 있다. 다음은 프로덕션-급의 완전한 구현이다.

### Tool 함수 정의 (전체 구현)

```python
import json
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# ============================================================================
# [Tool 1] 웹 검색 도구 (Mock 데이터)
# ============================================================================

@dataclass
class SearchResult:
    """검색 결과를 구조화하는 클래스"""
    title: str
    summary: str
    source: str

    def __str__(self):
        return f"{self.title}\n요약: {self.summary}\n출처: {self.source}"


def web_search(query: str) -> str:
    """
    인터넷에서 주어진 쿼리를 검색하고 상위 결과를 반환한다.

    실제 환경에서는 Google Custom Search API, Bing Search API, 또는
    Firecrawl 같은 웹 스크래핑 API를 호출할 수 있다.

    Args:
        query (str): 검색 키워드
                    예: "2024년 한국 평균 기온"
                    예: "삼성전자 주가"

    Returns:
        str: 검색 결과 요약 (상위 1~2개 결과)

    예시:
        >>> web_search("한국 GDP 2024")
        "1. 한국의 2024년 GDP는 약 1.45조 달러로 세계 12위입니다.
         2. IMF 발표: 한국의 올해 경제 성장률은 예상 2.3%에서 2.1%로 하향 조정되었습니다."
    """

    # Mock 웹 검색 데이터베이스 (실제로는 API 호출)
    web_database = {
        # 기온 관련
        "2024년 한국 평균 기온": {
            "title": "2024년 한국 평균 기온 통계",
            "summary": "기상청 발표: 2024년 한국 평균 기온은 14.8°C였으며, 이는 최근 30년 평균(12.8°C)보다 2.0°C 높다. 1973년 관측 이래 두 번째로 높은 기온을 기록했다.",
            "source": "기상청 (Korea Meteorological Administration)"
        },
        "2023년 한국 평균 기온": {
            "title": "2023년 한국 평균 기온 통계",
            "summary": "기상청 발표: 2023년 한국 평균 기온은 13.5°C였으며, 이는 최근 30년 평균보다 0.7°C 높다. 서울, 부산, 대구 모두 역대 높은 기온을 기록했다.",
            "source": "기상청"
        },

        # GDP 관련
        "한국 GDP": {
            "title": "한국의 2024년 GDP 통계",
            "summary": "한국의 2024년 명목 GDP는 약 1.45조 달러로 세계 12위이다. 전년도 대비 2.8% 증가했으며, 반도체 수출 호조가 주요 원인이다.",
            "source": "한국은행 (Bank of Korea)"
        },
        "일본 GDP": {
            "title": "일본의 2024년 GDP 통계",
            "summary": "일본의 2024년 명목 GDP는 약 4.23조 달러로 세계 3위이다. 엔화 약세와 수출 증가로 전년도 대비 1.9% 증가했다.",
            "source": "일본 재무성 (Ministry of Finance Japan)"
        },

        # 주가 관련
        "애플 주가": {
            "title": "AAPL (애플) 현재 주가",
            "summary": "현재 애플 주가: $182.50 (전일 대비 +2.3%). 최근 AI 칩 탑재로 기관 투자자들의 관심이 높다.",
            "source": "Yahoo Finance"
        },
        "삼성전자 주가": {
            "title": "Samsung Electronics (삼성전자) 현재 주가",
            "summary": "현재 삼성전자 주가: 65,500원 (전일 대비 -0.8%). HBM 메모리칩 수급 개선으로 점진적 회복세 예상.",
            "source": "한국거래소 (KRX)"
        },
    }

    # 쿼리와 키를 매칭하는 로직
    query_lower = query.lower()

    # 정확한 매칭 우선
    for key, data in web_database.items():
        if key.lower() in query_lower or query_lower in key.lower():
            result = SearchResult(
                title=data["title"],
                summary=data["summary"],
                source=data["source"]
            )
            return str(result)

    # 부분 매칭
    keywords = query_lower.split()
    for key, data in web_database.items():
        key_lower = key.lower()
        if any(kw in key_lower for kw in keywords):
            result = SearchResult(
                title=data["title"],
                summary=data["summary"],
                source=data["source"]
            )
            return str(result)

    # 결과 없음
    return f"검색 결과 없음: '{query}'에 대한 정보를 찾을 수 없습니다. 다른 검색어를 시도해 주세요."


# ============================================================================
# [Tool 2] 계산기 도구
# ============================================================================

def calculator(expression: str) -> float:
    """
    산술 수식을 계산한다.

    지원하는 연산: +, -, *, /, //, %, ** (괄호 포함)
    안전성: eval()의 위험성을 최소화하기 위해 허용된 문자만 검증한다.

    Args:
        expression (str): 계산할 수식
                         예: "14.8 - 13.5"
                         예: "(1200000000 - 1000000000) / 1000000000 * 100"

    Returns:
        float: 계산 결과
        str: 오류 메시지 (예외 발생 시)

    예시:
        >>> calculator("14.8 - 13.5")
        1.3

        >>> calculator("(150 - 140) / 140 * 100")
        7.142857142857143
    """

    try:
        # 안전한 계산을 위해 허용된 문자만 검증
        allowed_chars = set("0123456789.+-*/()")

        # 수식이 비어있지 않은지 확인
        if not expression or not expression.strip():
            return "오류: 빈 수식입니다."

        # 허용되지 않는 문자 검사
        invalid_chars = set(expression) - allowed_chars
        if invalid_chars:
            return f"오류: 허용되지 않는 문자가 포함되었습니다: {invalid_chars}"

        # 괄호 균형 검사
        if expression.count('(') != expression.count(')'):
            return "오류: 괄호가 균형맞지 않습니다."

        # eval 함수로 계산 (안전성 조치 후)
        result = eval(expression)

        # 결과가 숫자인지 확인
        if not isinstance(result, (int, float)):
            return f"오류: 계산 결과가 숫자가 아닙니다 ({type(result).__name__})"

        return float(result)

    except ZeroDivisionError:
        return "오류: 0으로 나눌 수 없습니다."
    except ValueError as e:
        return f"오류: 잘못된 수식 형식입니다 ({str(e)})"
    except Exception as e:
        return f"계산 오류: {str(e)}"


# ============================================================================
# [Tool 3] 데이터베이스 조회 도구
# ============================================================================

class ProductSalesDB:
    """
    회사 판매 데이터를 관리하는 데이터베이스 (Mock)

    구조: {(연도, 지역): {상품명: 매출액}}
    """

    def __init__(self):
        self.data = {
            (2024, "한국"): {
                "스마트폰": 1200000000,      # 12억
                "태블릿": 350000000,         # 3.5억
                "노트북": 450000000,         # 4.5억
                "이어폰": 200000000,         # 2억
            },
            (2023, "한국"): {
                "스마트폰": 1000000000,      # 10억
                "태블릿": 300000000,         # 3억
                "노트북": 400000000,         # 4억
                "이어폰": 150000000,         # 1.5억
            },
            (2024, "일본"): {
                "스마트폰": 850000000,       # 8.5억
                "태블릿": 280000000,         # 2.8억
                "노트북": 320000000,         # 3.2억
                "이어폰": 120000000,         # 1.2억
            },
            (2023, "일본"): {
                "스마트폰": 750000000,       # 7.5억
                "태블릿": 250000000,         # 2.5억
                "노트북": 280000000,         # 2.8억
                "이어폰": 100000000,         # 1억
            },
            (2024, "미국"): {
                "스마트폰": 2100000000,      # 21억
                "태블릿": 450000000,         # 4.5억
                "노트북": 650000000,         # 6.5억
                "이어폰": 300000000,         # 3억
            },
            (2023, "미국"): {
                "스마트폰": 1900000000,      # 19억
                "태블릿": 400000000,         # 4억
                "노트북": 600000000,         # 6억
                "이어폰": 250000000,         # 2.5억
            },
        }

    def query(self, year: int, region: str) -> Dict[str, int]:
        """
        특정 연도와 지역의 판매 데이터를 조회한다.

        Args:
            year (int): 조회 연도 (예: 2024, 2023)
            region (str): 지역 (예: "한국", "일본", "미국")

        Returns:
            dict: {상품명: 매출액} 형식의 딕셔너리
                  또는 오류 메시지

        예시:
            >>> db.query(2024, "한국")
            {"스마트폰": 1200000000, "태블릿": 350000000, ...}
        """

        # 입력 유효성 검사
        if not isinstance(year, int):
            return {"오류": f"연도는 정수여야 합니다 (입력: {year})"}

        if year < 2020 or year > 2025:
            return {"오류": f"지원하는 연도: 2023~2024 (입력: {year})"}

        if not isinstance(region, str):
            return {"오류": f"지역은 문자열이어야 합니다 (입력: {region})"}

        # 데이터 조회
        key = (year, region)
        if key in self.data:
            return self.data[key]
        else:
            available_regions = set(k[1] for k in self.data.keys() if k[0] == year)
            return {
                "오류": f"{year}년 {region} 판매 데이터를 찾을 수 없습니다.",
                "가능한_지역": list(available_regions)
            }


def query_sales_db(year: int, region: str) -> Dict[str, Any]:
    """
    회사 판매 데이터베이스를 조회한다.

    여러 상품의 판매 실적을 지역과 연도 기준으로 조회할 수 있다.

    Args:
        year (int): 조회 연도. 예: 2024, 2023
        region (str): 지역. 예: "한국", "일본", "미국"

    Returns:
        dict: {상품명: 매출액(원)} 형식의 딕셔너리
              또는 {"오류": "설명"} 형태의 오류 딕셔너리

    예시:
        >>> query_sales_db(2024, "한국")
        {
            "스마트폰": 1200000000,
            "태블릿": 350000000,
            "노트북": 450000000,
            "이어폰": 200000000
        }
    """

    db = ProductSalesDB()
    return db.query(year, region)


# ============================================================================
# [Agent State 정의]
# ============================================================================

from typing import TypedDict, Annotated, Optional, List

class AgentState(TypedDict):
    """
    ReAct Agent의 상태를 정의한다.

    Agent는 이 State 객체를 유지하면서, 각 반복 단계에서
    사용자 입력 → LLM 실행 → Tool 호출 → 상태 업데이트 과정을 거친다.
    """

    # [입력/출력]
    input: str
    """사용자의 질문 또는 요청"""

    output: Optional[str]
    """최종 답변 (Agent가 생성)"""

    # [대화 기록]
    messages: Annotated[List[dict], "LLM과의 주고받음 기록"]
    """
    각 단계의 메시지 기록
    형식: [{"role": "user|assistant|tool_results", "content": "..."}, ...]
    """

    # [Tool 실행 결과]
    tool_results: Annotated[Dict[str, Any], "각 도구의 실행 결과 저장"]
    """
    {"tool_name": result, ...} 형식으로 도구 실행 결과를 누적 저장한다.
    예: {"web_search": "결과1", "calculator": "1.3", "query_sales_db": {...}}
    """

    # [반복 제어]
    iterations: int
    """지금까지 몇 번의 THOUGHT-ACTION 루프를 돌았는가"""

    max_iterations: int
    """최대 허용 반복 횟수 (무한 루프 방지)"""

    # [디버깅 정보]
    debug_logs: Annotated[List[str], "디버깅용 로그 (선택)"]
    """각 단계에서의 상세 로그를 기록하여 Agent의 사고 과정을 추적한다"""


# ============================================================================
# [검증 및 테스트]
# ============================================================================

def test_tools_individually():
    """각 Tool을 단독으로 테스트하여 동작을 확인한다"""

    print("="*70)
    print("[Tool 검증] 각 도구의 기본 동작 확인")
    print("="*70)

    # Tool 1: 웹 검색
    print("\n[Tool 1] web_search 테스트")
    print("-" * 70)
    test_queries = [
        "2024년 한국 평균 기온",
        "한국 GDP",
        "없는 정보"
    ]
    for query in test_queries:
        result = web_search(query)
        print(f"질문: {query}")
        print(f"결과: {result[:100]}...\n" if len(result) > 100 else f"결과: {result}\n")

    # Tool 2: 계산기
    print("\n[Tool 2] calculator 테스트")
    print("-" * 70)
    test_expressions = [
        "14.8 - 13.5",
        "(1200000000 - 1000000000) / 1000000000 * 100",
        "0.5 * 2",
        "1 / 0"  # 오류 케이스
    ]
    for expr in test_expressions:
        result = calculator(expr)
        print(f"식: {expr}")
        print(f"결과: {result}\n")

    # Tool 3: 데이터베이스 조회
    print("\n[Tool 3] query_sales_db 테스트")
    print("-" * 70)
    test_queries = [
        (2024, "한국"),
        (2023, "일본"),
        (2024, "유럽")  # 오류 케이스
    ]
    for year, region in test_queries:
        result = query_sales_db(year, region)
        print(f"조회: {year}년 {region}")
        print(f"결과: {result}\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    test_tools_individually()
```

### 핵심 포인트: Tool 설계의 원칙

#### 1. **명확한 Docstring**

좋은 Tool의 Docstring은 4가지를 포함한다:
- **목적**: "이 Tool이 무엇을 하는가?"
- **파라미터**: "어떤 입력을 받고, 어떤 형식인가?"
- **반환값**: "어떤 형식의 결과를 주는가?"
- **예시**: "실제 사용 예시는?"

LLM (Agent)이 이 정보를 읽고 Tool을 올바르게 호출할지를 판단한다.

#### 2. **Mock 데이터의 현실성**

Tool의 Mock 데이터는 실제 데이터와 최대한 비슷해야 한다. 그래야 Agent가 실제 환경에서 Tool을 효과적으로 사용할 때를 대비할 수 있다.

예:
- web_search: 실제 검색 결과처럼 제목, 요약, 출처 구조
- calculator: 실제 계산처럼 다양한 연산자 지원, 오류 처리
- query_sales_db: 실제 데이터베이스처럼 다중 인덱스, 오류 시나리오

#### 3. **오류 처리의 명확성**

Tool이 실패할 때, 다음 두 가지 중 하나를 반환해야 한다:
- **구조화된 오류**: 딕셔너리나 객체로 오류 정보 제공 (Agent가 파싱 가능)
- **자연어 오류**: 사람이 읽을 수 있는 오류 메시지

```python
# 좋은 예: 구조화된 오류
return {
    "오류": "지원하지 않는 지역",
    "가능한_지역": ["한국", "일본", "미국"]
}

# 나쁜 예: 디버깅 정보 노출
return f"KeyError: ({year}, {region}) not in database at line 234"
```

#### 4. **입력 유효성 검사**

Tool은 받은 입력을 즉시 검사해야 한다. 이렇게 하면 Agent가 오류 원인을 빠르게 파악할 수 있다.

```python
# 유효성 검사 단계
1. 입력이 비어있거나 None인가?
2. 입력의 타입이 맞는가? (str, int, dict 등)
3. 입력의 범위가 유효한가? (year: 2020~2025)
4. 필수 키가 모두 있는가? (dict인 경우)
```

### 흔한 실수

#### 실수 1: Tool이 부분적으로만 Mock 데이터를 제공

```python
# 틀림
def web_search(query: str) -> str:
    if "한국 기온" in query:
        return "한국의 평균 기온은 12°C입니다"
    else:
        return "알 수 없음"

# 이 경우 다른 query에 대해 실패율이 높다
```

```python
# 맞음
def web_search(query: str) -> str:
    # Mock 데이터베이스에 50개 이상의 항목 포함
    web_database = {
        "2024년 한국 기온": {...},
        "한국 GDP": {...},
        "애플 주가": {...},
        ...
    }
    # 정확 매칭 → 부분 매칭 → 결과 없음 순서로 처리
```

#### 실수 2: Tool의 반환 타입이 일관성 없음

```python
# 틀림
def query_sales_db(year, region):
    if year == 2024:
        return {"스마트폰": 1200000000}  # dict
    else:
        return "데이터 없음"  # str
    # Agent가 반환값 타입을 예측할 수 없다

# 맞음
def query_sales_db(year, region):
    if year == 2024:
        return {"스마트폰": 1200000000}  # dict
    else:
        return {"오류": "데이터 없음"}  # 여전히 dict
    # 항상 dict 형식이므로 Agent가 처리하기 쉽다
```

#### 실수 3: Tool 함수 이름이 모호함

```python
# 틀림
def search(q):  # 이름이 불명확
    pass

def calc(e):  # 축약형 사용
    pass

# 맞음
def web_search(query: str) -> str:  # 명확하고 구체적
    pass

def calculator(expression: str) -> float:  # 목적이 분명
    pass
```

---

## 체크포인트 2 모범 구현: ReAct 루프 + LangGraph 구성

### 조건부 라우팅 함수

```python
def should_continue(state: AgentState) -> str:
    """
    현재 상태를 분석하여 다음 Node를 결정한다.

    Returns:
        str: 다음 step
             - "llm": LLM을 다시 실행
             - "tools": Tool을 실행
             - "end": Agent 종료
    """

    # [조건 1] 최대 반복 횟수 도달
    if state["iterations"] >= state["max_iterations"]:
        print(f"[INFO] 최대 반복 횟수({state['max_iterations']}) 도달 → 종료")
        return "end"

    # [조건 2] 메시지가 없음 (초기 상태)
    if not state["messages"]:
        print("[INFO] 초기 상태 → LLM 실행")
        return "llm"

    # [조건 3] 최근 메시지 분석
    latest_message = state["messages"][-1]

    # 역할 확인
    role = latest_message.get("role", "")
    content = latest_message.get("content", "")

    # assistant 메시지를 분석하여 다음 단계 결정
    if role == "assistant":
        # [종료 신호] "최종 답변:", "ANSWER:" 포함
        if "최종 답변:" in content or "ANSWER:" in content:
            # 상태에 최종 답변 저장
            state["output"] = content.replace("최종 답변:", "").replace("ANSWER:", "").strip()
            print(f"[INFO] 최종 답변 생성 → 종료")
            return "end"

        # [Tool 호출 신호] "<tool_call>" 포함
        elif "<tool_call>" in content:
            print(f"[INFO] Tool 호출 감지 → Tool 실행 Node로 이동")
            return "tools"

        # [재LLM 신호] 그 외의 경우
        else:
            print(f"[INFO] 중간 단계 → LLM 재실행")
            return "llm"

    # [기본값] Tool 실행 결과 메시지면 LLM 재실행
    elif role == "tool_results":
        print(f"[INFO] Tool 결과 수신 → LLM 재실행")
        return "llm"

    # [기본값] 알 수 없는 역할
    else:
        print(f"[INFO] 알 수 없는 역할 '{role}' → LLM 실행")
        return "llm"
```

### LLM 실행 Node (규칙 기반 시뮬레이션)

```python
def run_llm_node(state: AgentState) -> AgentState:
    """
    LLM을 실행하여 다음 행동을 결정한다.

    실제 환경에서는 ChatOpenAI, Claude 등 LLM API를 호출하지만,
    여기서는 규칙 기반 시뮬레이션으로 Agent의 동작을 보여준다.

    Returns:
        AgentState: 업데이트된 상태
    """

    print(f"\n{'='*70}")
    print(f"[반복 {state['iterations'] + 1}] LLM Node 실행")
    print(f"{'='*70}")

    # [Step 1] 현재 상태 분석
    if not state["messages"]:
        # 초기 상태: 사용자 입력만 있음
        current_input = state["input"]
        print(f"[분석] 초기 상태: '{current_input}'")
    else:
        # 진행 중: 최근 메시지까지의 컨텍스트 분석
        current_input = state["messages"][-1]["content"]
        print(f"[분석] 최근 메시지: {current_input[:100]}...")

    # [Step 2] 도구 결과가 있으면 최종 답변 생성
    if state["tool_results"]:
        print(f"[도구 결과] {len(state['tool_results'])}개 도구 결과 확인")

        # 도구 결과 텍스트로 변환
        tool_results_text = "\n".join([
            f"  · {tool}: {str(result)[:100]}"
            for tool, result in state["tool_results"].items()
        ])

        thought = f"""[THOUGHT]
이제 충분한 정보를 모았습니다. 도구 결과를 분석하여 최종 답변을 생성하겠습니다.

도구 실행 결과:
{tool_results_text}
"""

        final_answer = f"""[ANSWER]
질문: {state['input']}

분석 결과:
{state['tool_results']}

최종 답변: [도구 결과를 바탕으로 한 종합 답변]
"""

        state["messages"].append({
            "role": "assistant",
            "content": final_answer
        })

        print(f"[응답]\n{final_answer[:200]}...")

    else:
        # [Step 3] 사용자 질문 분석 및 필요한 도구 결정
        query = state["input"].lower()

        # 패턴 매칭으로 필요한 도구 결정
        tools_needed = []
        thought_text = f"[THOUGHT]\n질문: '{state['input']}'\n\n필요한 정보와 도구 분석:\n"

        # 기온 관련
        if any(word in query for word in ["기온", "온도", "날씨"]):
            tools_needed.append(("web_search", "2024년 한국 평균 기온"))
            tools_needed.append(("web_search", "2023년 한국 평균 기온"))
            thought_text += "  1. 웹 검색으로 2024년과 2023년 기온 데이터 조회\n"
            thought_text += "  2. 계산 도구로 기온 차이 계산\n"

        # GDP 관련
        elif "gdp" in query or "국내총생산" in query:
            tools_needed.append(("web_search", "한국 GDP"))
            tools_needed.append(("web_search", "일본 GDP"))
            thought_text += "  1. 웹 검색으로 한국 GDP 조회\n"
            thought_text += "  2. 웹 검색으로 일본 GDP 조회\n"
            thought_text += "  3. 계산 도구로 비율 비교\n"

        # 주가 관련
        elif "주가" in query or "주식" in query:
            if "애플" in query:
                tools_needed.append(("web_search", "애플 주가"))
            if "삼성" in query or "삼성전자" in query:
                tools_needed.append(("web_search", "삼성전자 주가"))
            thought_text += "  1. 웹 검색으로 관련 주가 정보 조회\n"

        # 매출 관련
        elif "매출" in query or "판매" in query or "수익" in query:
            tools_needed.append(("query_sales_db", (2024, "한국")))
            tools_needed.append(("query_sales_db", (2023, "한국")))
            thought_text += "  1. 데이터베이스에서 2024년 한국 판매 데이터 조회\n"
            thought_text += "  2. 데이터베이스에서 2023년 한국 판매 데이터 조회\n"
            thought_text += "  3. 합산 및 비교 계산\n"

        # 도구가 결정되지 않으면 일반 웹 검색
        if not tools_needed:
            tools_needed.append(("web_search", query))
            thought_text += f"  1. 웹 검색으로 '{query}' 관련 정보 조회\n"

        # [Step 4] Tool 호출 생성
        tool_calls = ""
        for i, (tool_name, tool_input) in enumerate(tools_needed, 1):
            if tool_name == "web_search":
                tool_calls += f"<tool_call>{{'tool': 'web_search', 'input': {{'query': '{tool_input}'}}}}</tool_call>\n"
            elif tool_name == "query_sales_db":
                year, region = tool_input
                tool_calls += f"<tool_call>{{'tool': 'query_sales_db', 'input': {{'year': {year}, 'region': '{region}'}}}}</tool_call>\n"
            elif tool_name == "calculator":
                tool_calls += f"<tool_call>{{'tool': 'calculator', 'input': {{'expression': '{tool_input}'}}}}</tool_call>\n"

        action_text = f"""[ACTION]
{tool_calls}"""

        state["messages"].append({
            "role": "assistant",
            "content": thought_text + action_text
        })

        print(f"[응답]\n{thought_text}")
        print(f"{action_text[:150]}...")

    # [Step 5] 상태 업데이트
    state["iterations"] += 1

    return state
```

### Tool 실행 Node

```python
def execute_tools_node(state: AgentState) -> AgentState:
    """
    LLM이 요청한 도구들을 실제로 실행한다.

    Returns:
        AgentState: 도구 결과가 추가된 상태
    """

    print(f"\n{'─'*70}")
    print(f"[Tool 실행 Node]")
    print(f"{'─'*70}")

    # [Step 1] 최근 LLM 응답에서 Tool 호출 추출
    latest_response = state["messages"][-1].get("content", "")

    # <tool_call>...</tool_call> 형식의 Tool 호출을 정규식으로 추출
    import re
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', latest_response, re.DOTALL)

    print(f"[감지] {len(matches)}개의 Tool 호출 감지")

    # [Step 2] 각 Tool 호출 파싱 및 실행
    for match_idx, match in enumerate(matches, 1):
        try:
            # JSON 파싱
            tool_json = json.loads(match)
            tool_name = tool_json.get("tool", "")
            tool_input = tool_json.get("input", {})

            print(f"\n  [{match_idx}] Tool: {tool_name}")
            print(f"      Input: {tool_input}")

            # 각 Tool 호출 및 실행
            if tool_name == "web_search":
                query = tool_input.get("query", "")
                result = web_search(query)
                tool_key = f"web_search_{match_idx}"

            elif tool_name == "calculator":
                expression = tool_input.get("expression", "")
                result = calculator(expression)
                tool_key = f"calculator_{match_idx}"

            elif tool_name == "query_sales_db":
                year = tool_input.get("year", 2024)
                region = tool_input.get("region", "한국")
                result = query_sales_db(year, region)
                tool_key = f"query_sales_db_{match_idx}"

            else:
                result = f"알 수 없는 도구: {tool_name}"
                tool_key = f"unknown_{match_idx}"

            # 결과 저장
            state["tool_results"][tool_key] = result

            # 결과 로깅
            result_str = str(result)
            if len(result_str) > 100:
                print(f"      → {result_str[:100]}...")
            else:
                print(f"      → {result_str}")

        except json.JSONDecodeError as e:
            print(f"  [{match_idx}] JSON 파싱 오류: {e}")
            state["tool_results"][f"error_{match_idx}"] = f"JSON 파싱 오류: {e}"

        except Exception as e:
            print(f"  [{match_idx}] 실행 오류: {e}")
            state["tool_results"][f"error_{match_idx}"] = f"Tool 실행 오류: {e}"

    # [Step 3] Tool 결과를 메시지에 추가
    state["messages"].append({
        "role": "tool_results",
        "content": str(state["tool_results"])
    })

    print(f"\n[저장] {len(state['tool_results'])}개의 Tool 결과를 State에 저장")

    return state
```

### StateGraph 구성

```python
from langgraph.graph import StateGraph, END

def build_agent_graph():
    """
    ReAct Agent의 StateGraph를 구성하고 컴파일한다.

    그래프 구조:
        시작 → LLM Node → [조건부 라우팅]
                          ├→ Tool Node → LLM Node (반복)
                          └→ END (종료)

    Returns:
        compiled_graph: 실행 가능한 컴파일된 그래프
    """

    # [Step 1] StateGraph 생성
    graph = StateGraph(AgentState)

    print("[그래프 구성] StateGraph 생성")

    # [Step 2] Node 추가
    graph.add_node("llm", run_llm_node)
    graph.add_node("tools", execute_tools_node)

    print("[그래프 구성] 2개의 Node 추가: llm, tools")

    # [Step 3] 시작점 설정
    graph.set_entry_point("llm")

    print("[그래프 구성] Entry Point 설정: llm")

    # [Step 4] Edge 추가
    # 4-1. LLM Node에서의 조건부 라우팅
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",  # Tool 호출 필요 → tools Node
            "llm": "llm",      # 재LLM 필요 → llm Node (반복)
            "end": END         # 종료 → 그래프 종료
        }
    )

    # 4-2. Tool Node 실행 후 LLM으로
    graph.add_edge("tools", "llm")

    print("[그래프 구성] Edge 추가:")
    print("   - llm → [조건부] tools, llm, END")
    print("   - tools → llm")

    # [Step 5] 컴파일
    agent = graph.compile()

    print("[그래프 구성] StateGraph 컴파일 완료")

    return agent
```

### 핵심 포인트: ReAct 루프의 효율성

#### 1. **Tool 호출 순서의 중요성**

```python
# 비효율적인 순서
순서: 계산 → 웹 검색 → 계산
문제: 데이터가 없는데 계산하려고 시도

# 효율적인 순서
순서: 웹 검색 → 계산 → 추가 검색 (필요하면)
이유: 데이터를 먼저 수집한 뒤 분석한다
```

#### 2. **반복 횟수 최소화**

```python
# 최대 반복: 5회
반복 1: web_search (2024 데이터) → 성공
반복 2: web_search (2023 데이터) → 성공
반복 3: calculator (14.8 - 13.5) → 성공
반복 4: [최종 답변] → 종료

# 효율: 4회 반복으로 완료 (최대 5회 중)
```

#### 3. **오류 복구 전략**

```python
# 오류 시나리오
반복 1: web_search("2024년 한국 기온") → 결과 없음
반복 2: THOUGHT: "더 일반적 검색어로 재시도"
        web_search("한국 평균 기온") → 성공
반복 3: [최종 답변]

# 자동 재시도로 오류를 극복한다
```

### 흔한 실수

#### 실수 1: should_continue가 항상 같은 값을 반환

```python
# 틀림
def should_continue(state):
    if "<tool_call>" in state["messages"][-1]["content"]:
        return "tools"
    return "llm"  # 항상 llm으로 돌아간다 → 무한 루프 위험

# 맞음
def should_continue(state):
    if state["iterations"] >= state["max_iterations"]:
        return "end"  # 명시적 종료 조건

    if "<tool_call>" in ...:
        return "tools"
    elif "최종 답변:" in ...:
        return "end"  # 명시적 종료 조건
    else:
        return "llm"
```

#### 실수 2: Tool 결과를 state에 누적하지 않음

```python
# 틀림
state["tool_results"] = {tool_name: result}  # 이전 결과를 덮어씀

# 맞음
state["tool_results"].update({tool_name: result})  # 누적
# 또는
state["tool_results"][tool_name] = result  # 누적
```

#### 실수 3: Tool 호출 JSON이 불완전함

```python
# 틀림
tool_call = f"<tool_call>{{'tool': 'web_search', 'input': {{'query': '{query}'}}}}</tool_call>"
# 큰따옴표가 섞여서 JSON 파싱 실패

# 맞음
tool_call = f'<tool_call>{{"tool": "web_search", "input": {{"query": "{query}"}}}}</tool_call>'
# 또는
import json
tool_json = json.dumps({"tool": "web_search", "input": {"query": query}})
tool_call = f"<tool_call>{tool_json}</tool_call>"
```

---

## 체크포인트 3 모범 구현: Agent 테스트 + ReAct 루프 분석

### Agent 초기화 및 실행

```python
def initialize_agent_state(user_input: str, max_iterations: int = 10) -> AgentState:
    """
    Agent의 초기 상태를 설정한다.

    Args:
        user_input (str): 사용자의 질문
        max_iterations (int): 최대 반복 횟수 (기본값 10)

    Returns:
        AgentState: 초기화된 Agent 상태
    """

    return {
        "input": user_input,
        "output": None,
        "messages": [{"role": "user", "content": user_input}],
        "tool_results": {},
        "iterations": 0,
        "max_iterations": max_iterations,
        "debug_logs": [f"[초기화] 사용자 입력: '{user_input}'"]
    }


def run_agent_with_logging(agent, user_input: str, max_iterations: int = 10) -> Dict[str, Any]:
    """
    Agent를 실행하고 전체 과정을 로깅한다.

    Args:
        agent: 컴파일된 LangGraph Agent
        user_input (str): 사용자의 질문
        max_iterations (int): 최대 반복 횟수

    Returns:
        dict: Agent 실행 결과 (최종 답변, 도구 사용 기록, 반복 횟수 등)
    """

    print(f"\n{'='*70}")
    print(f"[질문] {user_input}")
    print(f"{'='*70}")

    # 초기 상태 설정
    initial_state = initialize_agent_state(user_input, max_iterations)

    try:
        # Agent 실행
        result = agent.invoke(initial_state)

        # 결과 요약
        print(f"\n{'─'*70}")
        print(f"[최종 결과]")
        print(f"{'─'*70}")
        print(f"반복 횟수: {result['iterations']} / {result['max_iterations']}")
        print(f"사용된 도구: {', '.join(result['tool_results'].keys()) if result['tool_results'] else '없음'}")
        print(f"최종 답변: {result.get('output', '답변 없음')[:150]}...")

        return result

    except Exception as e:
        print(f"\n[오류] Agent 실행 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return initial_state
```

### 테스트 질문 3가지 + 실행

```python
def test_agent():
    """
    Agent를 여러 테스트 질문으로 검증한다.

    테스트 3가지:
    1. 웹 검색 + 계산 (기온 비교)
    2. 다중 웹 검색 + 계산 (GDP 비교)
    3. 데이터베이스 조회 + 계산 (매출 분석)
    """

    # Agent 구성
    print("\n[Agent 생성]")
    agent = build_agent_graph()

    # 테스트 질문
    test_queries = [
        {
            "input": "2024년 한국 평균 기온은 몇 도였고, 2023년보다 몇 도 올랐나?",
            "expected_tools": ["web_search", "calculator"],
            "expected_iterations": 3
        },
        {
            "input": "한국 GDP와 일본 GDP를 비교하면 어느 쪽이 더 큰가?",
            "expected_tools": ["web_search", "calculator"],
            "expected_iterations": 3
        },
        {
            "input": "2024년 한국의 전체 매출은 얼마인가? (스마트폰+태블릿+노트북+이어폰 합산)",
            "expected_tools": ["query_sales_db", "calculator"],
            "expected_iterations": 3
        }
    ]

    # 각 질문 실행
    results = []

    for query_info in test_queries:
        result = run_agent_with_logging(
            agent,
            query_info["input"],
            max_iterations=10
        )
        results.append({
            "query": query_info["input"],
            "result": result,
            "expected": query_info
        })

    return results


def analyze_react_loop(result: AgentState, query: str):
    """
    각 질문의 ReAct 루프 과정을 상세히 분석한다.

    Args:
        result (AgentState): Agent 실행 결과
        query (str): 원래 질문
    """

    print(f"\n{'='*70}")
    print(f"[ReAct 루프 분석] {query}")
    print(f"{'='*70}")

    # [분석 1] 메시지 흐름
    print(f"\n[메시지 흐름] 총 {len(result['messages'])}개 메시지")
    print(f"{'─'*70}")

    for i, msg in enumerate(result["messages"], 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # 역할 별로 다르게 출력
        if role == "user":
            print(f"[{i}] 사용자 질문: {content[:80]}")

        elif role == "assistant":
            # THOUGHT-ACTION 추출
            if "[THOUGHT]" in content:
                thought_part = content.split("[ACTION]")[0] if "[ACTION]" in content else content
                print(f"[{i}] 어시스턴트 (THOUGHT): {thought_part[:100]}...")
            else:
                print(f"[{i}] 어시스턴트: {content[:100]}...")

        elif role == "tool_results":
            print(f"[{i}] Tool 결과: {content[:100]}...")

    # [분석 2] 도구 사용 분석
    print(f"\n[도구 사용] {len(result['tool_results'])}개 도구 호출")
    print(f"{'─'*70}")

    for tool_name, tool_result in result["tool_results"].items():
        result_str = str(tool_result)[:80]
        print(f"  {tool_name}: {result_str}...")

    # [분석 3] 효율성 분석
    print(f"\n[효율성 분석]")
    print(f"{'─'*70}")
    print(f"  · 반복 횟수: {result['iterations']} / {result['max_iterations']}")
    print(f"  · 효율성: {(result['iterations'] / result['max_iterations'] * 100):.1f}%")
    print(f"  · Tool 사용 수: {len(result['tool_results'])}")

    # [분석 4] 최종 답변
    print(f"\n[최종 답변]")
    print(f"{'─'*70}")
    if result.get("output"):
        print(f"  {result['output'][:200]}...")
    else:
        print(f"  답변 없음")


# ============================================================================
# Main: 테스트 실행
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ReAct Agent 테스트 시작")
    print("="*70)

    # Agent 테스트
    results = test_agent()

    # 각 결과 분석
    for query_result in results:
        analyze_react_loop(query_result["result"], query_result["query"])

    print("\n" + "="*70)
    print("테스트 완료")
    print("="*70)
```

### 예상 실행 결과

```
======================================================================
ReAct Agent 테스트 시작
======================================================================

[그래프 구성] StateGraph 생성
[그래프 구성] 2개의 Node 추가: llm, tools
[그래프 구성] Entry Point 설정: llm
[그래프 구성] Edge 추가:
   - llm → [조건부] tools, llm, END
   - tools → llm
[그래프 구성] StateGraph 컴파일 완료

======================================================================
[질문] 2024년 한국 평균 기온은 몇 도였고, 2023년보다 몇 도 올랐나?
======================================================================

======================================================================
[반복 1] LLM Node 실행
======================================================================
[분석] 초기 상태: '2024년 한국 평균 기온은 몇 도였고, 2023년보다 몇 도 올랐나?'
[응답]
[THOUGHT]
질문: '2024년 한국 평균 기온은 몇 도였고, 2023년보다 몇 도 올랐나?'

필요한 정보와 도구 분석:
  1. 웹 검색으로 2024년과 2023년 기온 데이터 조회
  2. 계산 도구로 기온 차이 계산

[ACTION]
<tool_call>{"tool": "web_search", "input": {"query": "2024년 한국 평균 기온"}}</tool_call>
<tool_call>{"tool": "web_search", "input": {"query": "2023년 한국 평균 기온"}}</tool_call>
...

──────────────────────────────────────────────────────────────────────
[Tool 실행 Node]
──────────────────────────────────────────────────────────────────────
[감지] 2개의 Tool 호출 감지

  [1] Tool: web_search
      Input: {'query': '2024년 한국 평균 기온'}
      → 2024년 한국 평균 기온 통계
기상청 발표: 2024년 한국 평균 기온은 14.8°C였으며, ...

  [2] Tool: web_search
      Input: {'query': '2023년 한국 평균 기온'}
      → 2023년 한국 평균 기온 통계
기상청 발표: 2023년 한국 평균 기온은 13.5°C였으며, ...

[저장] 2개의 Tool 결과를 State에 저장

======================================================================
[반복 2] LLM Node 실행
======================================================================
[분석] 도구 결과 확인
[응답]
[THOUGHT]
이제 충분한 정보를 모았습니다. 도구 결과를 분석하여 최종 답변을 생성하겠습니다.

[ANSWER]
질문: 2024년 한국 평균 기온은 몇 도였고, 2023년보다 몇 도 올랐나?

최종 답변: 2024년 한국 평균 기온은 14.8°C였으며, 2023년(13.5°C)보다 1.3°C 올랐습니다.

======================================================================
[최종 결과]
======================================================================
반복 횟수: 2 / 10
사용된 도구: web_search_1, web_search_2
최종 답변: 2024년 한국 평균 기온은 14.8°C였으며, 2023년(13.5°C)보다 1.3°C 올랐습니다.

[ReAct 루프 분석] 2024년 한국 평균 기온은 몇 도였고, 2023년보다 몇 도 올랐나?
======================================================================

[메시지 흐름] 총 4개 메시지
──────────────────────────────────────────────────────────────────────
[1] 사용자 질문: 2024년 한국 평균 기온은 몇 도였고, 2023년보다 몇 도 올랐나?
[2] 어시스턴트 (THOUGHT): [THOUGHT]
질문: '2024년 한국 평균 기온은 몇 도였고, 2023년보다 몇 도 올랐나?'...
[3] Tool 결과: {'web_search_1': '2024년 한국 평균 기온 통계...', 'web_search_2': '2023년 한국 평균 기온...'}
[4] 어시스턴트: [ANSWER]
질문: 2024년 한국 평균 기온은 몇 도였고, 2023년보다 몇 도 올랐나?...

[도구 사용] 2개 도구 호출
──────────────────────────────────────────────────────────────────────
  web_search_1: 2024년 한국 평균 기온 통계
기상청 발표: 2024년 한국 평균 기온은...
  web_search_2: 2023년 한국 평균 기온 통계
기상청 발표: 2023년 한국 평균 기온은...

[효율성 분석]
──────────────────────────────────────────────────────────────────────
  · 반복 횟수: 2 / 10
  · 효율성: 20.0%
  · Tool 사용 수: 2
```

### 핵심 포인트: ReAct 루프의 투명성

ReAct 루프의 가장 큰 장점은 **투명성**이다. Agent가 어떻게 생각하고 어떤 도구를 썼는지 명확히 보인다.

#### 좋은 ReAct 루프의 특징

```
THOUGHT: "질문을 분석했고, 필요한 정보와 도구를 파악했다"
         ↓
ACTION:  "도구 이름, 파라미터를 JSON 형식으로 명확히 지정"
         ↓
OBSERVATION: "도구 실행 결과를 수신"
         ↓
(반복) 또는 ANSWER: "최종 답변 생성"

각 단계가 명확하므로:
1. Agent의 오류를 쉽게 찾을 수 있다
2. 도구 선택의 타당성을 검증할 수 있다
3. 사용자에게 답변의 근거를 명확히 보여줄 수 있다
```

#### 나쁜 ReAct 루프의 특징

```
(불명확한 생각과정)
     ↓
(도구 호출 형식이 불규칙)
     ↓
(도구 결과가 누적되지 않음)
     ↓
(최종 답변이 근거 없음)

이 경우:
1. 오류 발생 원인 파악 어려움
2. Agent의 판단 검증 불가능
3. 사용자가 답변을 신뢰하기 어려움
```

### 흔한 실수

#### 실수 1: Tool 결과를 무시하고 환각(hallucination) 답변 생성

```python
# 틀림
OBSERVATION: web_search → "한국 GDP는 1.45조 달러"
ANSWER: "한국 GDP는 2조 달러입니다" # 도구 결과와 무관한 답변!

# 맞음
OBSERVATION: web_search → "한국 GDP는 1.45조 달러"
ANSWER: "한국 GDP는 1.45조 달러입니다" # 도구 결과를 반영
```

#### 실수 2: 반복 횟수가 과다함

```python
# 비효율: 반복이 10회 이상 (최대 10회)
반복 1: web_search (첫 검색)
반복 2: web_search (재검색)
반복 3: web_search (또 재검색)
...
반복 10: 최종 답변
# 각 단계에서 "더 나은 도구가 있을까?" 계속 찾는 중

# 효율: 반복이 2~3회
반복 1: web_search (필요한 검색)
반복 2: [최종 답변 생성]
# 필요한 정보를 얻으면 바로 답변
```

#### 실수 3: Tool 호출이 중복됨

```python
# 비효율
반복 1: web_search("한국 기온")
반복 2: web_search("한국 기온") # 같은 검색을 다시!
반복 3: web_search("한국 기온") # 또 반복!

# 효율: Tool 결과를 재사용
반복 1: web_search("한국 기온")
반복 2: [이미 받은 결과를 활용해 최종 답변]
```

---

## 참고 코드 파일

다음 파일에서 전체 구현을 확인할 수 있다:

- **practice/chapter12/code/12-1-react-agent.py** — 전체 Tool + Agent 구현
- **practice/chapter12/code/12-2-agent-with-error-handling.py** — Error Handling 및 재시도 로직
- **practice/chapter12/code/12-3-multi-tool-composition.py** — 여러 Tool의 조합 예제

### 코드 실행 방법

```bash
# 가상환경 활성화
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# 각 실습 실행
python practice/chapter12/code/12-1-react-agent.py
python practice/chapter12/code/12-2-agent-with-error-handling.py
python practice/chapter12/code/12-3-multi-tool-composition.py
```

---

## 최종 학습 정리

### 12주차 핵심 개념 요약

1. **AI Agent와 Chatbot의 차이**: Agent는 도구를 조합하여 복잡한 문제를 해결하는 자율적 시스템이다.

2. **ReAct 루프의 4단계**: THOUGHT (분석) → ACTION (도구 호출) → OBSERVATION (결과 수신) → (반복 또는 종료)

3. **Tool의 역할**: Agent의 "팔과 눈" 역할을 하며, LLM의 한계(knowledge cutoff, 계산 정확성 등)를 보완한다.

4. **Tool 설계의 원칙**: 명확한 Docstring, Mock 데이터의 현실성, 오류 처리, 입력 유효성 검사

5. **LangGraph의 가치**: State, Node, Edge를 이용하여 Agent의 복잡한 로직을 시각적이고 재사용 가능한 형태로 구성한다.

6. **Tool Composition**: 여러 도구를 순차적/병렬적으로 조합하여 단일 도구로는 불가능한 문제를 해결한다.

7. **Error Handling 전략**: 도구 실행 오류 시 재시도, 대체 도구 사용, 입력 수정 등 다양한 대응 방식

8. **ReAct의 투명성**: Agent의 사고 과정(THOUGHT), 행동(ACTION), 관찰(OBSERVATION)이 명확하게 기록되어 오류 대응과 신뢰도 향상이 용이하다.

### 다음 학습으로의 연결

- **13주차 A회차**: **상황 인식 Agent (Situation Aware Agent)** — 더 많은 도구, 더 복잡한 문제 상황, 사전 정보(context) 활용
- **13주차 B회차**: 실제 비즈니스 시나리오 (고객 지원 Agent, 데이터 분석 Agent)에 LangGraph Agent 적용

이 12주차의 ReAct Agent는 LLM 시대의 AI 엔지니어링의 핵심 기술이다. 올바른 Tool 설계, 명확한 ReAct 루프, 견고한 Error Handling을 통해, 단순한 Chatbot을 넘어 자율적이고 신뢰할 수 있는 AI 시스템을 구축할 수 있다.

---

**생성일**: 2026-02-25
**대상 독자**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
**난이도**: 중상급 (파이썬, 딥러닝 기초, LangChain 개념 선수)
