# MS Word 변환 시스템

Markdown 원고를 전문적인 MS Word 문서(.docx)로 변환하는 시스템입니다.

---

## 요구사항

- Node.js 18+ (LTS 권장)
- npm 9+

## 설치

```bash
cd ms-word
npm install
```

## 사용법

### 개별 챕터 변환

```bash
npm run convert:chapter 2    # 2장만 변환
npm run convert:chapter 5    # 5장만 변환
```

### 전체 챕터 일괄 변환

```bash
npm run convert:all
```

### 통합 도서 생성

```bash
npm run create:book
```

## 출력 파일

- **개별 챕터**: `output/ch{N}.docx`
- **통합 도서**: `output/{project}-complete-book.docx`

## 디렉토리 구조

```
ms-word/
├── config/
│   ├── book-metadata.json       # 도서 메타데이터
│   └── formatting-standards.js  # 편집 기준
├── src/
│   ├── converter.js             # 핵심 변환 엔진
│   ├── convert-chapter.js       # 개별 챕터 변환
│   ├── convert-all.js           # 일괄 변환
│   └── create-book.js           # 통합 도서 생성
├── output/                      # 생성된 Word 파일
├── templates/
│   ├── preface.md               # 머리말 템플릿
│   ├── references.md            # 참고문헌 템플릿
│   └── author-bio.md            # 저자 소개 템플릿
├── package.json
└── README.md
```

## 설정 파일

### book-metadata.json

```json
{
  "title": "도서 제목",
  "subtitle": "부제",
  "author": "저자명",
  "publisher": "출판사",
  "year": 2026,
  "chapters": [
    { "number": 1, "title": "1장 제목" },
    { "number": 2, "title": "2장 제목" }
  ]
}
```

### formatting-standards.js

페이지 설정, 폰트, 스타일 등 편집 기준을 정의합니다.

## 워크플로우

```
docs/ch{N}.md (Markdown)
        │
        ▼
    [변환 엔진]
        │
        ▼
ms-word/output/ch{N}.docx
```

## 주요 기능

1. **Markdown → Word 변환**
   - 제목, 본문, 코드블록, 표 변환
   - 이미지 삽입
   - 참고문헌 형식화

2. **스타일 적용**
   - 학술 문서 형식
   - 페이지 번호
   - 머리말/바닥글

3. **통합 도서 생성**
   - 표지
   - 목차 자동 생성
   - 표/그림 목차
   - 참고문헌 통합

## 문제 해결

### 한글 폰트 깨짐

Windows에서는 맑은 고딕, macOS에서는 Apple SD Gothic Neo를 기본으로 사용합니다.

### 이미지 경로 오류

이미지 경로는 `docs/` 폴더 기준 상대 경로로 지정하세요.

---

**버전**: 1.0
