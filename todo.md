# 작업 체크리스트

## Phase 0: 구 강의계획서(2025) 기반 집필 — ✅ 완료 (아카이브 대상)

- [x] 프로젝트 초기 구조 설정 (2026-01-03)
- [x] 구 강의계획서 기반 전 챕터 집필 (ch1-14, 13개 콘텐츠)
- [x] contents.md 전면 개편 — 신 강의계획서(2026) 전환 (2026-02-08)
- [x] ch3 신 강의계획서 맞춤 재작성 + Multi-LLM 리뷰 (2026-02-08)
- [x] Notion 발행 스크립트 작성 (2026-02-08)

---

## Phase 1: 구버전 아카이브 — ✅ 완료 (2026-02-11)

- [x] docs/ 구버전 12개 → `_archive/old-syllabus/docs/`
- [x] schema/ 구버전 13개 → `_archive/old-syllabus/schema/`
- [x] content/drafts/ 구버전 12개 → `_archive/old-syllabus/content/drafts/`
- [x] content/research/ 구버전 12개 → `_archive/old-syllabus/content/research/`
- [x] content/graphics/ 구버전 12개 디렉토리 → `_archive/old-syllabus/content/graphics/`
- [x] content/reviews/ 구버전 12개 → `_archive/old-syllabus/content/reviews/`
- [x] practice/ 구버전 12개 디렉토리 → `_archive/old-syllabus/practice/`
- [x] ms-word/ → `_archive/old-syllabus/ms-word/`
- [x] ch3 자산은 현재 위치 유지 (신버전)

---

## Phase 2: 신 강의계획서 기반 재집필 — ⏳ 진행 중 (6/13 완료)

각 챕터당 7단계 워크플로우(Planning → Research → Analysis → Implementation → Optimization → QA → 발행)를 수행한다.

### 소폭수정 챕터 (3개)

구버전 문서를 기반으로 신 강의계획서 형식에 맞게 수정한다.

#### ch1: AI 시대의 개막과 개발 환경 구축 — ✅ 완료 (2026-02-11)
- [x] 집필계획서 (schema/chap1.md)
- [x] 리서치 (구버전 재활용 + 2024-2026 NLP 동향 조사)
- [x] 실습 코드 (practice/chapter1/code/ — 3개 파일, 실행 결과 확인)
- [x] 그래픽 (content/graphics/ch1/ — Mermaid 2개)
- [x] Multi-LLM 리뷰 (GPT-4o 평균 8.7 + Grok-4 평균 9.3)
- [x] 최종 원고 (docs/ch1.md — 610줄)
> **변경사항**: 구 ch1 재활용 + 3교시제 + 1.3 Tensor/Autograd 신규 + 1.4 Copilot 실습 신규

#### ch8: 토픽 모델링
- [ ] 집필계획서
- [ ] 리서치
- [ ] 초안
- [ ] 실습 코드
- [ ] 그래픽
- [ ] Multi-LLM 리뷰
- [ ] 최종 원고
> **참고**: 구 ch8 (토픽 모델링) 내용 유사, 3교시제 형식 적용

#### ch14: 최종 프로젝트 개발
- [x] 집필계획서 (schema 기반)
- [x] 리서치 (contents.md 사양 참고)
- [x] ch14A 최종 원고 (docs/ch14A.md)
- [x] ch14B 최종 원고 (docs/ch14B.md — 658줄, 2026-02-25)
- [x] ch14C (A회차, B회차 발표 후 작성)
- [-] 실습 코드 (개인 프로젝트이므로 표준화된 코드 불필요)
- [-] 그래픽 (학생이 자신의 프로젝트에 맞게 작성)
- [-] Multi-LLM 리뷰 (개인 프로젝트 가이드이므로 불필요)
> **변경사항**: 구 ch14 프로젝트 구조 참고 + 신 강의계획서 3교시제 형식 (A: 계획, B: 개발+피드백, C: 발표) + 학생 개별 지도 중심

### 재작성 챕터 (9개)

구버전 자산을 참고하되 신 강의계획서에 맞게 새로 작성한다.

#### ch2: 딥러닝 핵심 원리와 PyTorch 실전 — ✅ 완료 (2026-02-11)
- [x] 집필계획서 (schema/chap2.md)
- [x] 리서치 (구 ch4 재활용 + 신경망 기초 신규)
- [x] 실습 코드 (practice/chapter2/code/ — 3개 파일, 실행 결과 확인)
- [x] 그래픽 (content/graphics/ch2/ — Mermaid 3개)
- [x] Multi-LLM 리뷰 (GPT-4o 평균 8.5 + Grok-4 평균 8.3)
- [x] 최종 원고 (docs/ch2.md — 718줄)
> **변경사항**: 구 ch4 재활용 + 1교시 신경망 기초 신규 + 3교시제 + BoW 텍스트 분류 실습

#### ch4: Transformer 아키텍처 심층 분석 — ✅ 완료 (2026-02-11)
- [x] 집필계획서 (schema/chap4.md)
- [x] 리서치 (구 ch6 참조 + Transformer 구성 요소 분석)
- [x] 실습 코드 (practice/chapter4/code/ — 3개 파일, 실행 결과 확인)
- [x] 그래픽 (content/graphics/ch4/ — Mermaid 4개)
- [x] Multi-LLM 리뷰 (GPT-4o 평균 8.3 + Grok-4 평균 8.2)
- [x] 최종 원고 (docs/ch4.md — 633줄)
> **변경사항**: 구 ch6 참조 + 3교시제 + Decoder 구현 + Tokenization 심화 + BPE 밑바닥 구현

#### ch5: LLM 아키텍처: BERT와 GPT — ✅ 완료 (2026-02-12)
- [x] 집필계획서 (schema/chap5.md)
- [x] 리서치 (구 ch9 BERT + 구 ch10 GPT 참조)
- [x] 실습 코드 (practice/chapter5/code/ — 3개 파일, 실행 결과 확인)
- [x] 그래픽 (content/graphics/ch5/ — Mermaid 4개)
- [x] Multi-LLM 리뷰 (GPT-4o 평균 9.2 + Grok-4 평균 8.3)
- [x] 최종 원고 (docs/ch5.md — 698줄)
> **변경사항**: 구 ch9 (BERT) + 구 ch10 (GPT) 합쳐서 압축 + 3교시제 + Hugging Face 실전 섹션 신규

#### ch6: LLM API 활용과 프롬프트 엔지니어링 — ✅ 완료 (2026-02-23)
- [x] 집필계획서 (schema/chap6.md)
- [x] 리서치 (content/research/ch6-research.md — API SDK 패턴, 참고문헌 검증)
- [x] 실습 코드 (practice/chapter6/code/ — 3개 파일, 실행 결과 확인)
- [x] 그래픽 (content/graphics/ch6/ — Mermaid 4개)
- [x] Multi-LLM 리뷰 (Claude Haiku 4.5 평균 8.8/10, GPT-4o/Grok-4 미설정)
- [x] 최종 원고 (docs/ch6.md — 655줄)
> **변경사항**: 구 ch13 프롬프트 참고 + API/Function Calling/Structured Output/LLM 평가 신규 + 캐시 폴백

#### ch9: LLM 파인튜닝 (1) — Full Fine-tuning
- [ ] 집필계획서
- [ ] 리서치
- [ ] 초안
- [ ] 실습 코드
- [ ] 그래픽
- [ ] Multi-LLM 리뷰
- [ ] 최종 원고
> **참고 자산**: 구 ch11 (LLM 파인튜닝 1 — Full FT) 재구성

#### ch10: LLM 파인튜닝 (2) — PEFT와 LoRA
- [ ] 집필계획서
- [ ] 리서치
- [ ] 초안
- [ ] 실습 코드
- [ ] 그래픽
- [ ] Multi-LLM 리뷰
- [ ] 최종 원고
> **참고 자산**: 구 ch12 (LLM 파인튜닝 2 — PEFT) 재구성

#### ch11: RAG 시스템 구축
- [ ] 집필계획서
- [ ] 리서치
- [ ] 초안
- [ ] 실습 코드
- [ ] 그래픽
- [ ] Multi-LLM 리뷰
- [ ] 최종 원고
> **참고 자산**: 구 ch13 일부 (RAG 부분) 분리 확장

#### ch12: AI Agent 개발
- [ ] 집필계획서
- [ ] 리서치
- [ ] 초안
- [ ] 실습 코드
- [ ] 그래픽
- [ ] Multi-LLM 리뷰
- [ ] 최종 원고
> **참고 자산**: 없음 — **완전 신규 작성**

#### ch15: 최종 평가
- [x] 집필계획서 (schema 기반, contents.md 15주차 참고)
- [x] ch15A: 개인 프로젝트 최종 발표 강의자료 (docs/ch15A.md — 766줄, 2026-02-25)
  - 발표 운영 가이드 (5단계 시간 배분, 질의응답 대비)
  - 평가 루브릭 (7개 영역, 100점 만점 배점)
  - 슬라이드 템플릿 (10~15장 권장)
  - 동료 평가 양식
- [-] ch15B: 기말고사 (강의 중 준비)
- [-] ch15C: (발표 후 평가 종합)
- [-] 실습 코드 (개인 프로젝트이므로 표준화된 코드 불필요)
- [-] 그래픽 (학생이 자신의 프로젝트에 맞게 작성)
- [-] Multi-LLM 리뷰 (시험 + 발표 평가 가이드이므로 불필요)
> **변경사항**: 구 ch14 프로젝트 발표 자료 재활용 + 신 강의계획서 최종 발표 평가 기준 신규 정의 + 전문적 발표 운영 가이드 신규 추가

#### ch13: 모델 배포와 프로덕션
- [ ] 집필계획서
- [ ] 리서치
- [ ] 초안
- [ ] 실습 코드
- [ ] 그래픽
- [ ] Multi-LLM 리뷰
- [ ] 최종 원고
> **참고 자산**: 없음 — **완전 신규 작성**

---

## Phase 3: 발행 — 미착수

### 3-1. Notion 발행
- [ ] Notion Integration 생성 및 API Key 발급
- [ ] Notion 데이터베이스 생성 (Title, Chapter 속성)
- [ ] .env 파일 설정
- [ ] `--contents` 목차 발행 테스트
- [ ] ch1~ch14 전체 발행 (ch7 시험 제외)

---

## Phase 4: 후속 작업

- [ ] checklists/book-progress.md 실제 데이터로 업데이트
- [ ] 표지 및 머리말/서문 작성
- [ ] 색인(Index) 생성
- [ ] 최종 교정 및 통합 검수
- [ ] 참고문헌 검증 (URL/DOI 확인)
- [ ] 실습 코드 크로스 플랫폼 검증 (macOS + Windows)

---

## 범례

| 기호 | 의미 |
|------|------|
| [x] | 완료 |
| [ ] | 미완료 |
| [-] | 해당 없음 |

---

**마지막 업데이트**: 2026-02-25
**현재 Phase**: Phase 2 진행 중 (ch1, ch2, ch3, ch4, ch5, ch6, ch14, ch15A 완료 = 8개 챕터)
**다음 작업**: ch8 소폭수정 (토픽 모델링) 또는 ch9 재작성 (LLM 파인튜닝 1) 또는 ch10 재작성 (LoRA/QLoRA)
