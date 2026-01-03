# AGENTS.md (repo-wide)

이 저장소에서 작업할 때는 반드시 `CLAUDE.md`의 원칙을 최우선으로 따른다. 이 파일은 그 내용을 "실제 작업 흐름" 기준으로 실행 가능하게 정리한 것이다.

## 0) 최우선 준수 문서
- `CLAUDE.md` (필수): 분량/스타일, 코드 제시 원칙, "실제 실행 결과만" 사용, 크로스 플랫폼 호환, 참고문헌 검증, 폴더 구조
- `docs/sample.md` (필수): 본문 서술·표·핵심 코드(3~5줄) 제시 스타일 표준
- `contents.md` (필수): 각 장의 절 구성/집필 방향/강화 항목(요구사항)

## 0.5) 집필 대상 및 방향
- **대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
- **집필 목표**: 이론과 방법을 쉽고 명확하게 설명
- **분량 기준**: 최소 500줄 (내용에 따라 탄력 조정)
- **핵심 원칙**:
  - 복잡한 개념은 비유와 예시로 먼저 소개
  - 수학 공식은 직관적 해석을 반드시 동반
  - 전문 용어는 처음 등장 시 명확히 정의
  - 단계별 설명: 개념 → 원리 → 구현 → 응용
  - **그래픽 적극 활용**: 복잡한 개념은 Mermaid 다이어그램이나 시각자료로 설명

## 0.6) 그래픽 필수 활용 원칙
- **모든 복잡한 개념에 다이어그램 포함**: 알고리즘 구조, 데이터 흐름, 모델 아키텍처 등
- **권장 도구**: Mermaid (마크다운 내 텍스트 작성), Python matplotlib
- **저장 위치**: `content/graphics/ch{N}/`
- **본문 포함 형식**:
  ```markdown
  ```mermaid
  flowchart TD
      A[입력] --> B[처리] --> C[출력]
  ```
  **그림 N.M** 제목
  ```
- **그래픽 생성 시점**: 본문 작성과 동시에 (@writer + @graphic 협업)

## 1) 절대 금지 (하드 룰)
- 더미/가상 데이터로 결과를 "만들지" 않는다.
- "예시 출력입니다" 같은 가짜 실행 결과를 본문에 쓰지 않는다.
- 허구의 참고문헌/URL/DOI를 쓰지 않는다. 검증 가능한 실제 문헌만 사용한다.
- 플랫폼 종속 경로/명령어를 본문/코드에 하드코딩하지 않는다.

## 2) 본문(원고) 작성 원칙
- 본문에는 **핵심 코드만 3~5줄** 넣고, 전체 실행 코드는 `practice/`에 둔다.
- 본문에서 전체 코드는 항상 아래 형태로 참조한다:
  `_전체 코드는 practice/chapter{N}/code/{파일명}.py 참고_`
- 숫자/표/성능 지표/산출물 목록은 **실제 실행으로 얻은 것만** 기재한다.
- `contents.md`의 "절 구성/강화 항목"을 충족했는지 항상 교차 점검한다.
- **개조식 금지**: 본문에서 불릿 포인트를 피하고 서술형으로 작성 (예외: 표, 학습목표, 핵심정리)

## 3) 실습 코드 작성 원칙
- 실습 코드는 `practice/chapter{N}/code/` 아래에 **단독 실행 가능**해야 한다.
- 크로스 플랫폼 경로: `pathlib.Path` 또는 `os.path.join()` 사용, 상대 경로 우선.
- 기본 실행 형태:
  ```bash
  cd practice/chapter{N}
  python3 -m venv venv && source venv/bin/activate
  pip install -r code/requirements.txt
  python code/{N}-{M}-{주제}.py
  ```
- 환경에 따라 `python`이 없을 수 있으므로, `python3`을 기준으로 안내한다.

## 4) "실제 실행"과 산출물 관리
- 변경한 실습 스크립트는 최소 1회 실제 실행하여 오류 없이 동작함을 확인한다.
- 실습 산출물은 기본적으로 `practice/chapter{N}/data/output/` 아래로 저장한다.
- 문서(`docs/ch{N}.md`)에는 산출물 경로를 일관되게 적고, 실제 생성 여부를 확인한다.

## 5) Word 변환 (출판 파이프라인)
- Markdown 원고는 최종본을 `docs/ch{N}.md`에 둔다.
- Word 변환은 `ms-word` 파이프라인을 사용한다:
  ```bash
  cd ms-word && npm install    # 최초 1회
  npm run convert:chapter {N}  # 개별 챕터
  npm run convert:all          # 전체 일괄
  npm run create:book          # 통합 도서
  ```
- 변환 결과는 `ms-word/output/ch{N}.docx`에 생성된다.

## 6) LLM 품질 검증 (선택)
- Multi-LLM 리뷰 스크립트로 챕터 검토를 수행할 수 있다:
  ```bash
  cd scripts
  python3 multi_llm_review.py --chapter {N}
  ```
- 결과 JSON은 `content/reviews/ch{N}_review_YYYY-MM-DD.json`에 저장된다.
- 리뷰의 "주요 이슈(critical_issues)"는 가능한 범위에서 원고/코드에 반영한다.

## 7) 진행 체크리스트 업데이트
- 원고/코드/그래픽/검토 진행상태는 `checklists/book-progress.md`에 반영한다.
- 표의 범례(R/W/C/G/V)를 유지하고, 작업 완료 시 해당 칸을 갱신한다.

## 8) 작업 후 최소 확인 (출력 기준)
- 수정한 `docs/ch{N}.md`가 `contents.md` 요구사항을 충족하는지 확인
- 본문에 들어간 결과값/파일 경로가 실제 실행/생성 결과와 일치하는지 확인
- 필요한 경우 Word 변환(`ms-word/output/ch{N}.docx`)까지 갱신

---

## 에이전트 역할 요약

| 에이전트 | 역할 | 도구 | 출력 위치 |
|---------|------|------|----------|
| @planner | 집필계획서 작성 | **Plan Mode** | `schema/chap{N}.md` |
| @researcher | 자료 조사 | WebSearch, WebFetch | `content/research/` |
| @writer | 원고 초안 작성 | Read, Write, Edit | `content/drafts/` → `docs/` |
| @coder | 실습 코드 작성 | Write, Bash | `practice/chapter{N}/code/` |
| @reviewer | 품질 검토 | Read, Grep | `docs/ch{N}.md` (최종) |
| @graphic | 시각자료 제작 | Mermaid, Write | `content/graphics/` |

## Plan Mode 사용 (계획 단계)

**@planner 에이전트는 Claude Code의 Plan Mode를 사용합니다.**

### 워크플로우
```
1. EnterPlanMode 진입
   ↓
2. 탐색 (Glob, Grep, Read)
   - contents.md 분석
   - 기존 챕터 참조
   - 관련 자료 검토
   ↓
3. 집필계획서 작성
   - schema/chap{N}.md에 상세 계획 작성
   ↓
4. ExitPlanMode로 사용자 승인 요청
   ↓
5. 승인 후 다음 단계(@researcher) 진행
```

### 사용 시점
- 새 챕터 시작 시 (필수)
- 목차 변경 시 (영향 분석)
- 대규모 수정 시 (계획 수립)

---

**마지막 업데이트**: 2026-01-02
