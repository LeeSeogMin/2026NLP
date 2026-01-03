# 전문도서 작성 시스템

AI 에이전트 기반의 체계적인 전문도서 작성 템플릿입니다.

## 개요

이 시스템은 6개의 전문 에이전트가 협력하여 고품질 전문도서를 작성하는 워크플로우를 제공합니다.

## 6개 전문 에이전트

| 에이전트 | 역할 | 담당 영역 |
|---------|------|----------|
| Planner | 집필계획자 | 전체 기획, 목차 설계, 일정 관리 |
| Researcher | 리서처 | 자료 수집, 사실 확인, 참고문헌 |
| Writer | 작가 | 본문 작성, 문체 통일, 가독성 |
| Coder | 코드작성자 | 예제 코드, 실행 검증 |
| Reviewer | 검토자 | 품질 검증, 일관성 체크 |
| Graphic | 그래픽 | 다이어그램, 시각화 |

## 디렉토리 구조

```
book_research/
├── .claude/
│   ├── agents/              # 6개 에이전트 정의
│   └── commands/            # 슬래시 커맨드
├── docs/                    # 완성된 챕터 (ch01.md ~ ch16.md)
├── schema/                  # 챕터별 설계 문서
├── practice/                # 실습 코드
│   └── chapter{N}/
│       ├── code/            # 전체 소스코드
│       └── data/            # 샘플 데이터
├── content/
│   ├── research/            # 조사 자료
│   ├── drafts/              # 초안
│   ├── graphics/            # 그래픽 자료
│   └── reviews/             # 리뷰 결과
├── ms-word/                 # Word 변환 시스템
│   ├── config/              # 설정
│   ├── src/                 # 변환 스크립트
│   ├── output/              # 출력 파일
│   └── templates/           # 템플릿
├── checklists/              # 진행 체크리스트
├── contents.md              # 목차 템플릿
├── CLAUDE.md                # 프로젝트 컨텍스트
└── AGENTS.md                # 에이전트 운영 규칙
```

## 7단계 워크플로우

```
[1. 기획] → [2. 자료수집] → [3. 분석] → [4. 집필] → [5. 최적화] → [6. 검증] → [7. Word 변환]
```

## 시작하기

### 1. 목차 설정
`contents.md`에서 도서의 목차를 정의합니다.

### 2. 챕터 작성
각 챕터별로 에이전트가 순차적으로 작업합니다:
1. **Planner**: `schema/ch{N}_schema.md` 설계
2. **Researcher**: `content/research/ch{N}/` 자료 수집
3. **Writer**: `content/drafts/ch{N}/` 초안 작성
4. **Coder**: `practice/chapter{N}/code/` 코드 작성
5. **Reviewer**: `content/reviews/ch{N}_review.md` 검토
6. **Graphic**: 다이어그램 및 시각화 추가
7. 최종본을 `docs/ch{N}.md`로 이동

### 3. Word 변환
```bash
cd ms-word
npm install
npm run convert:all       # 전체 챕터 변환
npm run create:book       # 통합 도서 생성
```

## 작성 규칙

### 코드 표현
- 본문: 3-5줄 핵심 코드만
- 전체 코드: `practice/chapter{N}/code/`
- 모든 코드는 실제 실행 가능해야 함

### 수식 표현
- LaTeX 금지, Unicode 사용
- 예: x² + y² = r², ∑, ∫, α, β, γ

### 경로 규칙
- 크로스플랫폼: `pathlib.Path` 또는 `os.path.join()`
- 하드코딩 금지

## 진행 상황 추적

`checklists/book-progress.md`에서 각 챕터의 진행 상황을 관리합니다.

## 슬래시 커맨드

- `/prime` - 프로젝트 컨텍스트 로드

---

**버전**: 1.0
**기반**: geoAI 도서 작성 시스템
