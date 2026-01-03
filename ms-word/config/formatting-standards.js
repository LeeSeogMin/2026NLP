/**
 * MS Word 문서 편집 기준
 * 학술 도서 출판 표준에 맞춘 설정
 */

module.exports = {
  // 페이지 설정
  page: {
    size: {
      width: 210,  // mm (A4)
      height: 297  // mm (A4)
    },
    margins: {
      top: 30,     // mm
      bottom: 30,  // mm
      left: 25,    // mm
      right: 25    // mm
    },
    lineSpacing: 1.6  // 160%
  },

  // 폰트 설정
  fonts: {
    korean: {
      windows: 'Malgun Gothic',
      mac: 'Apple SD Gothic Neo',
      fallback: 'Noto Sans KR'
    },
    english: 'Times New Roman',
    code: 'Consolas'
  },

  // 제목 스타일
  headings: {
    chapter: {
      size: 28,
      bold: true,
      spacing: { before: 400, after: 200 }
    },
    h1: {
      size: 24,
      bold: true,
      spacing: { before: 300, after: 150 }
    },
    h2: {
      size: 18,
      bold: true,
      spacing: { before: 250, after: 120 }
    },
    h3: {
      size: 14,
      bold: true,
      spacing: { before: 200, after: 100 }
    },
    h4: {
      size: 12,
      bold: true,
      spacing: { before: 150, after: 80 }
    }
  },

  // 본문 스타일
  body: {
    size: 11,
    spacing: { before: 0, after: 120 },
    firstLineIndent: 10  // mm
  },

  // 코드 블록 스타일
  code: {
    size: 9,
    backgroundColor: '#f5f5f5',
    borderColor: '#e0e0e0',
    padding: 10  // pt
  },

  // 표 스타일
  table: {
    headerBackground: '#4472C4',
    headerTextColor: '#FFFFFF',
    borderColor: '#BFBFBF',
    alternateRowColor: '#F2F2F2'
  },

  // 그림/표 캡션
  caption: {
    size: 10,
    alignment: 'center',
    spacing: { before: 80, after: 120 }
  },

  // 머리말/바닥글
  headerFooter: {
    size: 9,
    color: '#808080'
  },

  // 목차 스타일
  toc: {
    chapter: { size: 12, bold: true, indent: 0 },
    section: { size: 11, bold: false, indent: 10 },
    subsection: { size: 10, bold: false, indent: 20 }
  }
};
