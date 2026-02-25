# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# 딥러닝 자연어처리 교재 집필 프로젝트

> **부제**: LLM 시대의 NLP 엔지니어링: 원리부터 배포까지
> **대상 독자**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
> **집필 목표**: LLM의 원리를 깊이 이해하고, 실무에서 모델을 설계·튜닝·배포할 수 있는 AI 엔지니어를 양성하되, 직관적 비유와 단계적 설명으로 학부생이 쉽게 따라올 수 있도록 함

## 외부 메모리 전략

작업 시작 전 반드시 아래 파일을 참조하여 현재 상태를 파악한다:

| 파일 | 역할 | 갱신 시점 |
|------|------|----------|
| `context.md` | 프로젝트 목표, 완료된 작업, 현재 상태, 다음 단계 | 주요 작업 완료 시 |
| `todo.md` | Phase별 작업 체크리스트 | 작업 완료/추가 시 |
| `contents.md` | 최종 목차 (15주차 구성) | 목차 변경 시 |

**CRITICAL**: 작업 완료 후 반드시 `context.md`와 `todo.md`를 업데이트한다.

---

## 빠른 시작 명령어

### Notion 발행 (수업자료 배포)
```bash
pip install notion-client              # 의존성 설치 (최초 1회)
python scripts/notion_publish.py --chapter 2   # 개별 챕터 발행
python scripts/notion_publish.py --all         # 전체 발행
python scripts/notion_publish.py --contents    # 목차 발행
python scripts/notion_publish.py --chapter 2 --dry-run  # 파싱 테스트
```

### Python 실습 환경 (통합 가상환경)
```bash
python -m venv venv
source venv/bin/activate               # macOS/Linux
# venv\Scripts\activate                # Windows
pip install -r requirements.txt        # 프로젝트 루트의 통합 requirements
python practice/chapter{N}/code/{N}-{M}-{주제}.py   # 실습 코드 실행
```

### 실습 도구: VS Code + GitHub Copilot
- **IDE**: VS Code 1.102+
- **AI 도구**: GitHub Copilot Pro (학생 무료 — GitHub Student Developer Pack)
- **Copilot 모드**: Copilot Chat (Agent 모드)
- **역할 구분**: Copilot은 코드의 "주 작성자"가 아니라 구현의 **가속 도구**
  - A회차: Copilot 없이 이론과 원리 학습 (교수 시연 시 선택 사용)
  - B회차: Copilot 활용 2인1조 실습 + 디버깅

---

## 프로젝트 개요

6개 전문 에이전트가 협력하여 강의계획서를 **학부생 대상 교재**로 집필·검증·변환하는 워크플로우 시스템입니다.

### 도서 정보
- **제목**: 딥러닝 자연어처리
- **대상**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
- **구성**: 15주차 × 주2회 = 30회차 (기초 4주 + LLM 2주 + 중간고사 + 실무 5주 + 개인 프로젝트 2주 + 기말고사)
- **수업 형태**: A회차(이론+시연, 90분) + B회차(실습+토론, 90분)
- **예상 분량**: 약 500페이지
- **최종 목차**: `contents.md` 참조

### 평가 방식
| 항목 | 비중 | 비고 |
|------|------|------|
| 중간고사 | 20% | 객관식 (1~6주차 범위) |
| 기말고사 | 20% | 객관식 (8~13주차 범위) |
| 주별 과제 | 30% | 2인1조 협업, 개별 제출, 약 10회 |
| 개인 프로젝트 | 30% | 14~15주차, 구현 + 배포 + 발표 |

- 결석 -2점, 지각 -1점 (총점 감점제)
- B회차 2인1조 실습은 협업으로 진행하되 제출은 개별로 한다

### 강의 구성 철학
- **기초는 빠르게, 실무는 깊게**: RNN은 개념만, Transformer부터 본격 구현
- **밑바닥 구현 → 프레임워크 활용**: 원리를 안 뒤에 도구를 쓴다
- **취업 직결 기술 우선**: Agent, 배포, 평가 등 현업 필수 스킬 포함
- **직관 먼저, 수식 다음**: 모든 개념은 "왜 필요한가"부터 시작하고, 비유와 시각화로 직관을 잡은 뒤 수학적 정의로 넘어간다

### 수업 운영 구조 (주2회 90분 A/B 체계)

매주 2회 수업을 진행하며, A회차(이론+시연)와 B회차(실습+토론)로 구분한다:

**A회차 (90분) — 이론 + 시연**

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단(퀴즈 1문항) | 사용 안 함 |
| 00:05~00:55 | 이론 강의 (직관적 비유 → 개념 → 수식 → 원리) | 사용 안 함 |
| 00:55~01:25 | 라이브 코딩 시연 (교수 주도, 핵심 코드 시연) | 교수 시연 시 선택 사용 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 | |
| 01:28~01:30 | Exit ticket (1문항) | |

**B회차 (90분) — 실습 + 토론**

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 조 편성 + 역할 배분 (조원 A/B) | |
| 00:05~00:10 | A회차 핵심 리캡 + 과제 스펙 재확인 | |
| 00:10~00:55 | 2인1조 Copilot 구현 (체크포인트 3회) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (조별 1부) | |
| 01:00~01:20 | 결과 토론 (구현 전략 비교·성능 차이 분석) | |
| 01:20~01:28 | 교수 피드백 + 핵심 정리 | |
| 01:28~01:30 | 다음 주 예고 | |

**A회차 문서 형식** (`ch{N}A.md`):
```
## N주차 A회차: 제목
> **미션**: 수업이 끝나면 ...할 수 있다
### 학습목표
### 수업 타임라인 (표)
### 오늘의 질문 + 빠른 진단
### 이론 강의
  #### 절 (직관적 비유 → 개념 → 원리)
### 라이브 코딩 시연
  > **라이브 코딩 시연**: 교수 시연 설명
### 핵심 정리 + B회차 과제 스펙
### Exit ticket
```

**B회차 문서 형식** (`ch{N}B.md`):
```
## N주차 B회차: 제목
> **미션**: ...를 구현하고 결과를 분석할 수 있다
### 과제 스펙 + 체크포인트
### 2인1조 실습 (체크포인트 3회)
  > **Copilot 활용**: Copilot 사용 가이드
  #### 체크포인트 1: ...
  #### 체크포인트 2: ...
  #### 체크포인트 3: ...
### 제출 안내 (Google Classroom)
### 결과 토론 가이드
### 교수 피드백 포인트
```

**C파일** (`ch{N}C.md`): 모범 구현 + 해설 (B회차 제출 후 공개)

**교수 전용 요소**:
- `> **라이브 코딩 시연**:` — A회차에서 교수가 직접 코딩하며 보여주는 시연
- `> **Copilot 활용**:` — B회차에서 Copilot을 어떻게 활용하는지 안내
- `> **강의 팁**:` — 수업 운영 조언
- `> **토론 가이드**:` — B회차 결과 토론 운영 방향

### 집필 철학
- **직관적 비유 병렬 제시**: 기술적 깊이를 유지하면서, 모든 핵심 개념에 일상적 비유를 병렬로 제공
- **실습 연계**: 이론을 바로 코드로 확인할 수 있도록 구성
- **학술적 문체 유지**: 쉬운 설명이되 격식체 문장 사용

---

## 최우선 원칙

### 1. 분량 기준 (탄력적 적용)
| 항목 | 기준 |
|------|------|
| 장 전체 | **최소 500줄** (내용에 따라 탄력 조정) |
| A4 페이지 | 약 30쪽 내외 |
| 이론:코드 | 70% : 30% |

**장 유형별 분량**:
- 기초 개념 장: 500-600줄 (이론:실습 = 70:30)
- 핵심 기술 장: 600-700줄 (이론:실습 = 60:40)
- 실습 중심 장: 500-600줄 (이론:실습 = 40:60)
- 심화/응용 장: 700줄 이상 가능 (내용에 따라 조정)

**분량보다 이해도**: 위 기준은 가이드일 뿐이며, 학부생이 이해하기 쉽게 설명하는 것이 최우선

### 2. 실제 실행 원칙
- 모든 코드는 실제 실행하여 결과 획득
- 더미/가상 데이터 금지
- "예시 출력입니다" 형태의 가상 결과 금지
- **크로스 플랫폼 호환성**: 모든 코드는 Windows와 macOS 모두에서 실행 가능해야 함
  - 경로 구분자는 `os.path.join()` 또는 `pathlib.Path` 사용
  - 플랫폼 특정 명령어 사용 금지
  - 경로 하드코딩 대신 상대 경로 또는 환경 변수 활용

### 3. 참고문헌 검증
- 허구의 참고문헌 절대 금지
- 모든 인용은 실재 검증된 문헌만
- URL/DOI 가능한 포함

### 4. 학부생 대상 집필 방향 (CRITICAL — 모든 집필의 최우선 원칙)

**핵심 원칙**: 이 교재의 독자는 딥러닝·NLP를 **처음** 배우는 학부 3~4학년생이다. 대학원 세미나 자료가 아니라, "처음 접한 학생이 혼자 읽어도 따라갈 수 있는 교재"를 목표로 한다. 분량이 길어지더라도 이해도를 우선한다.

#### 4.1 직관적 설명 원칙 (CRITICAL)
- **"왜 필요한가" 먼저, "어떻게 작동하는가" 다음**: 모든 개념은 동기(문제 상황)를 먼저 제시하고, 그 문제를 해결하기 위해 이 기법이 등장했음을 보여준 뒤, 구체적 작동 방식으로 넘어간다
- **직관적 비유 필수**: 모든 핵심 개념에 `**직관적 이해**:` 블록으로 일상적 비유를 먼저 제시한 뒤 기술적 설명으로 넘어간다
  - 예: Attention = "시험 공부할 때 밑줄 긋기", LoRA = "눈 수술 대신 안경 끼기", RAG = "오픈북 시험"
  - 비유는 학부생이 일상에서 즉시 떠올릴 수 있는 구체적 상황을 사용한다 (추상적 비유 금지)
- **수식은 직관적 해석을 반드시 동반**: 수식을 제시한 뒤 "이 수식이 말하는 것은 결국 ..."로 한 문장 해석을 붙인다. 수식만 던지고 넘어가지 않는다
- **전문 용어는 처음 등장 시 영문 병기 + 한 줄 정의**: 예) "어텐션(Attention)은 입력의 어떤 부분에 집중할지 결정하는 메커니즘이다"
- **단계별 설명 순서**: 동기(왜?) → 직관적 비유 → 핵심 개념 → 수식/원리 → 구현 → 실행 결과 → 해석/응용

#### 4.1.1 설명 밀도 원칙 (CRITICAL)
- **한 문단에 새로운 개념은 1개만**: 여러 개념을 한꺼번에 쏟아내지 않는다. 하나를 충분히 설명한 뒤 다음으로 넘어간다
- **"그래서 무엇이 달라지는가"를 반드시 포함**: 개념 설명 후 "이것이 없으면 어떤 문제가 생기는가 / 이것이 있으면 무엇이 좋아지는가"를 구체적 예시로 보여준다
- **비교와 대조 적극 활용**: 새 개념을 설명할 때 이미 배운 개념과 비교한다 (예: "RNN이 한 글자씩 순서대로 읽는 사람이라면, Transformer는 문장 전체를 한눈에 보는 사람이다")
- **구체적 숫자와 예시 우선**: 추상적 설명 대신 작은 예시(3×3 행렬, 4개 단어 문장 등)로 과정을 보여준 뒤 일반화한다
- **"쉽게 말해서" 블록 적극 사용**: 기술적 설명이 길어진 뒤 `> **쉽게 말해서**: ...` 블록으로 핵심을 한두 문장으로 재정리한다

#### 4.2 코드 제시 원칙
- **핵심 코드만 본문에 포함** (3-5줄)
- **전체 구현 코드는 별도 파일로 분리**: `practice/chapter{N}/code/`
- 파일명은 절 번호와 주제를 반영: `{N}-{M}-{주제}.py`
- 본문에서는 참조 형태로만 언급

#### 4.3 실행 결과와 해석 필수 (CRITICAL)
- **절대 금지**: 가상의 결과값, "예시 출력", 임의로 만든 숫자
- **필수**: 코드를 실제로 실행하여 얻은 결과만 본문에 포함
- 결과 해석 및 실무적 시사점 도출

#### 4.4 표준 참조 문서
**집필 시 반드시 참조**: `docs/ch3A.md` (A회차 표준), `docs/ch3B.md` (B회차 표준)
- practice 폴더 참조 형식: `_전체 코드는 practice/chapter{N}/code/{파일명}.py 참고_`

#### 4.5 그래픽 활용 원칙 (CRITICAL)

**핵심**: 복잡한 개념은 반드시 그림/다이어그램으로 시각화하여 학부생의 이해를 돕는다.

**사용 가능한 방법**:

1. **Mermaid 다이어그램 (권장)**
   - 마크다운 내 텍스트로 작성
   - 버전 관리 용이
   - GitHub, VS Code 등에서 바로 렌더링
   - 플로우차트, 시퀀스, 클래스, 상태 다이어그램 지원

2. **Python 시각화 (matplotlib/seaborn)**
   - 데이터 기반 차트/그래프
   - `practice/chapter{N}/code/`에서 생성
   - `content/graphics/ch{N}/`에 PNG로 저장

**그래픽 저장 위치**:
- Mermaid 소스: `content/graphics/ch{N}/fig-{N}-{순번}-{주제}.mmd`
- 렌더링 이미지: `content/graphics/ch{N}/fig-{N}-{순번}-{주제}.png`

**본문에서 참조 형식**:
```markdown
```mermaid
flowchart LR
    A[입력] --> B[처리] --> C[출력]
```

**그림 N.1** 데이터 처리 흐름
```

**그래픽 필수 사용 상황**:
- 알고리즘/모델 구조 설명 시
- 데이터 흐름/파이프라인 설명 시
- 개념 간 관계 설명 시
- 성능 비교 시 (차트)

---

## 폴더 구조

```
project/
├── CLAUDE.md              # 이 파일 - 프로젝트 컨텍스트
├── AGENTS.md              # 운영 규칙
├── contents.md            # 최종 목차 및 집필 방향
├── schema/                # 집필계획서
│   └── chap{N}.md
├── .claude/
│   └── agents/            # 서브에이전트 정의 (6개)
│       ├── planner.md     # 집필계획자
│       ├── researcher.md  # 리서처
│       ├── writer.md      # 작가
│       ├── coder.md       # 코드작성자
│       ├── reviewer.md    # 검토자
│       └── graphic.md     # 그래픽
├── content/
│   ├── research/          # 리서치 결과
│   ├── drafts/            # 원고 초안
│   ├── graphics/          # 다이어그램/시각자료
│   └── reviews/           # LLM 리뷰 결과
├── docs/                  # 최종 완성 원고 (검토 완료)
│   ├── ch3A.md            # 집필 표준 참조 문서 (3주차 A회차)
│   ├── ch{N}A.md          # N주차 A회차 (이론+시연)
│   ├── ch{N}B.md          # N주차 B회차 (실습+토론)
│   └── ch{N}C.md          # N주차 C파일 (모범 구현+해설)
├── practice/              # 실습 코드 및 데이터
│   └── chapter{N}/
│       ├── code/          # 실행 가능한 전체 코드
│       │   ├── {N}-{M}-{주제}.py
│       │   └── requirements.txt
│       └── data/          # 실제/가상 데이터
│           ├── input/
│           └── output/
├── checklists/            # 진행 체크리스트
├── scripts/               # 자동화 스크립트
└── _archive/              # 구버전 자산 보관 (old-syllabus/)
```

---

## 7단계 워크플로우

```
[1단계: Planning] ⭐ Claude Code Plan Mode 사용
    EnterPlanMode
        │
    @planner ── 집필계획서 작성 ──▶ schema/chap{N}.md
        │
    ExitPlanMode (사용자 승인)
        │
        ▼
[2단계: Information Gathering]
    @researcher ── 자료 조사 ──▶ content/research/
        │
        ▼
[3단계: Analysis]
    정보 구조화 및 핵심 통찰 추출
        │
        ▼
[4단계: Implementation & Documentation]
    ├── @coder ── 코드 우선 작성 ──▶ practice/chapter{N}/code/
    ├── @writer ── 결과 기반 문서화 ──▶ content/drafts/
    └── @graphic ── 시각자료 ──▶ content/graphics/
        │
        ▼
[5단계: Optimization]
    일관성 및 완성도 검증
        │
        ▼
[6단계: Quality Verification]
    @reviewer ── 품질 검토 ──▶ docs/ch{N}A.md, ch{N}B.md, ch{N}C.md (최종 완성본)
        │
        ▼
[7단계: 발행]
    └── Notion 발행 ──▶ Notion 페이지 (수업자료 배포)
```

### 필수 단계 (5-7단계는 자동 수행)

**CRITICAL**: 4단계 완료 후 5-6-7단계를 **자동으로 연속 수행**한다.

#### 작업 명령어 해석 기준
| 사용자 명령 | 수행 범위 | 비고 |
|---|---|---|
| "N주차 작성" | 1~7단계 전체 (A+B+C) | **모든 단계 자동 수행** (Notion 발행 포함) |
| "N주차 A회차 작성" | 1~7단계 (A회차만) | A회차 단독 제작 |
| "N주차 B회차 작성" | 1~7단계 (B+C) | B회차 + C파일 제작 |
| "N주차 검토" | 5~7단계 | 일관성 검증 + 품질 리뷰 + 발행 |
| "N주차 발행" | 7단계만 | docs/ch{N}A.md, ch{N}B.md, ch{N}C.md → Notion |

---

## 에이전트 라우팅 규칙

### @planner (집필계획자)
- **트리거**: "계획", "스키마", "집필계획서", "구성"
- **도구**: **Claude Code Plan Mode** (EnterPlanMode → 탐색/계획 → ExitPlanMode)
- **출력**: `schema/chap{N}.md`

### @researcher (리서처)
- **트리거**: "조사해줘", "리서치", "자료 찾아줘", "참고문헌"
- **출력**: `content/research/ch{N}-{주제}.md`

### @writer (작가)
- **트리거**: "작성해줘", "초안", "원고", "본문"
- **출력**: `content/drafts/ch{N}A.md`, `content/drafts/ch{N}B.md`

### @coder (코드작성자)
- **트리거**: "코드", "실습", "예제", "구현"
- **출력**: `practice/chapter{N}/code/`

### @reviewer (검토자)
- **트리거**: "검토", "리뷰", "피드백", "수정"
- **출력**: 인라인 피드백 또는 `docs/ch{N}A.md`, `docs/ch{N}B.md`, `docs/ch{N}C.md`

### @graphic (그래픽)
- **트리거**: "다이어그램", "그래픽", "플로우차트", "아키텍처"
- **출력**: `content/graphics/ch{N}/`

---

## 핵심 작업 규칙

### 문체 가이드 (기술적 깊이 + 직관적 이해)
- **문체**: 객관적, 논리적, 설명적 — 단, 학부생 눈높이에 맞춤
- **종결어미**: 격식체 평서문 ('이다', '한다', '보인다')
- **문장 구조**: 짧고 명확한 문장 선호, 개조식 금지 (예외: 표, 학습목표)
- **용어**: 전문 용어는 처음 등장 시 영문 병기 + 간단한 설명 (예: "어텐션(Attention)은 입력의 어떤 부분에 집중할지 결정하는 메커니즘이다")
- **직관적 비유 필수**: 핵심 개념마다 `**직관적 이해**:` 블록으로 비유를 먼저 제시한 뒤 기술적 설명에 들어간다. 비유는 학부생이 일상에서 쉽게 떠올릴 수 있는 것을 사용한다
- **기술적 깊이 유지**: 비유로 직관을 잡은 뒤에는 PyTorch 구현, 수학적 원리 등 CS/AI 전공 수준의 기술적 깊이를 반드시 포함한다

### 수식 표기 (Unicode 인라인)
```
✅ 올바름: Yᵢₜ = αᵢ + λₜ + δ·Dᵢₜ + εᵢₜ
❌ 금지: $Y_{it} = \alpha_i + \lambda_t$ (LaTeX)
```

### 코드 스타일
- Python 3.10+
- PEP 8 준수
- 한국어 주석/docstring
- 실제 실행 결과만 사용

### 참고문헌 형식
```
저자명. (연도). 논문제목. *저널명*. URL/DOI
```

### 표/그림 제목 형식
- **표 제목**: 표 위에 작성 (`**표 2.1** 제목`)
- **그림 제목**: 그림 아래에 작성 (`**그림 3.2** 제목`)
- **번호 체계**: `{장번호}.{순번}`

---

## 환경 변수 (.env)

```bash
# Notion 발행 (수업자료 배포)
NOTION_API_KEY=secret_xxx              # Notion Integration Token
NOTION_DATABASE_ID=xxx                 # 대상 데이터베이스 ID

# Multi-LLM 리뷰 (선택)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
XAI_API_KEY=...

# 웹 스크래핑 (선택)
FIRECRAWL_API_KEY=...
```

### Notion 설정 가이드

1. https://www.notion.so/my-integrations 에서 **Internal Integration** 생성
2. Notion에서 수업자료용 **데이터베이스** 생성 (Title, Chapter(Number) 속성)
3. 데이터베이스에 Integration 연결: **Share → Invite → Integration 선택**
4. `.env` 파일에 `NOTION_API_KEY`, `NOTION_DATABASE_ID` 설정
5. `python scripts/notion_publish.py --contents` 로 테스트

### 6단계 Multi-LLM Review 모델 설정

**CRITICAL**: 6단계 품질 검토 시 반드시 다음 모델을 사용한다:

| Provider | 모델 ID | 용도 |
|----------|---------|------|
| OpenAI | `gpt-4o` | 리뷰 #1 |
| xAI | `grok-4-1-fast-reasoning` | 리뷰 #2 |

API 호출 예시:
```bash
# OpenAI GPT-4o
curl -s https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model": "gpt-4o", ...}'

# xAI Grok-4
curl -s https://api.x.ai/v1/chat/completions \
  -H "Authorization: Bearer $XAI_API_KEY" \
  -d '{"model": "grok-4-1-fast-reasoning", ...}'
```

---

## 체크리스트 위치

진행 상황은 `checklists/book-progress.md`에서 추적합니다.

---

## 에이전트 상세 정의

각 에이전트의 상세 지침은 `.claude/agents/` 폴더에 정의되어 있다:

| 에이전트 | 정의 파일 | 핵심 역할 |
|---------|----------|----------|
| @planner | `.claude/agents/planner.md` | Plan Mode로 집필계획서 작성 |
| @researcher | `.claude/agents/researcher.md` | 검증된 자료 수집 및 문헌 조사 |
| @writer | `.claude/agents/writer.md` | 학부생 눈높이 원고 작성 |
| @coder | `.claude/agents/coder.md` | 크로스플랫폼 실습 코드 개발 |
| @reviewer | `.claude/agents/reviewer.md` | 품질 검증 및 Multi-LLM 리뷰 |
| @graphic | `.claude/agents/graphic.md` | Mermaid/matplotlib 시각자료 |

---

## Multi-LLM 품질 검증 (6단계)

리뷰 스크립트로 외부 LLM 검증을 수행할 수 있다:

```bash
cd scripts
python3 multi_llm_review.py --chapter {N}
```

검증 결과: `content/reviews/ch{N}_review_YYYY-MM-DD.json`

---

**마지막 업데이트**: 2026-02-25
**교과목**: 딥러닝 자연어처리 — LLM 시대의 NLP 엔지니어링
**수업 형태**: 주2회 90분 A/B 체계
**템플릿 버전**: 5.0
