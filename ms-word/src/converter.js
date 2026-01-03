/**
 * Markdown to Word 핵심 변환 엔진
 *
 * 사용법:
 *   const Converter = require('./converter');
 *   const converter = new Converter();
 *   await converter.convert('docs/ch01.md', 'output/ch01.docx');
 */

const fs = require('fs-extra');
const path = require('path');
const MarkdownIt = require('markdown-it');
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  Table,
  TableRow,
  TableCell,
  WidthType,
  AlignmentType,
  BorderStyle
} = require('docx');

const standards = require('../config/formatting-standards');

class Converter {
  constructor() {
    this.md = new MarkdownIt({
      html: true,
      breaks: true,
      linkify: true
    });
  }

  /**
   * Markdown 파일을 Word 문서로 변환
   */
  async convert(inputPath, outputPath) {
    console.log(`변환 시작: ${inputPath}`);

    // Markdown 읽기
    const markdown = await fs.readFile(inputPath, 'utf-8');

    // 파싱
    const tokens = this.md.parse(markdown, {});

    // Word 요소로 변환
    const children = this.tokensToDocx(tokens);

    // 문서 생성
    const doc = new Document({
      sections: [{
        properties: {
          page: {
            margin: {
              top: standards.page.margins.top * 56.7,  // mm to twip
              bottom: standards.page.margins.bottom * 56.7,
              left: standards.page.margins.left * 56.7,
              right: standards.page.margins.right * 56.7
            }
          }
        },
        children: children
      }]
    });

    // 저장
    const buffer = await Packer.toBuffer(doc);
    await fs.writeFile(outputPath, buffer);

    console.log(`변환 완료: ${outputPath}`);
    return outputPath;
  }

  /**
   * Markdown 토큰을 Word 요소로 변환
   */
  tokensToDocx(tokens) {
    const children = [];
    let i = 0;

    while (i < tokens.length) {
      const token = tokens[i];

      switch (token.type) {
        case 'heading_open':
          const level = parseInt(token.tag.slice(1));
          const headingContent = tokens[i + 1].content;
          children.push(this.createHeading(headingContent, level));
          i += 3;  // heading_open, inline, heading_close
          break;

        case 'paragraph_open':
          const paragraphContent = tokens[i + 1].content;
          if (paragraphContent) {
            children.push(this.createParagraph(paragraphContent));
          }
          i += 3;
          break;

        case 'fence':
          children.push(this.createCodeBlock(token.content, token.info));
          i += 1;
          break;

        case 'table_open':
          const tableTokens = [];
          while (tokens[i].type !== 'table_close') {
            tableTokens.push(tokens[i]);
            i++;
          }
          children.push(this.createTable(tableTokens));
          i++;
          break;

        case 'bullet_list_open':
        case 'ordered_list_open':
          // 리스트 처리 (간단 버전)
          while (tokens[i].type !== 'bullet_list_close' &&
                 tokens[i].type !== 'ordered_list_close') {
            if (tokens[i].type === 'inline') {
              children.push(this.createParagraph('• ' + tokens[i].content));
            }
            i++;
          }
          i++;
          break;

        default:
          i++;
      }
    }

    return children;
  }

  /**
   * 제목 생성
   */
  createHeading(text, level) {
    const levelMap = {
      1: HeadingLevel.HEADING_1,
      2: HeadingLevel.HEADING_2,
      3: HeadingLevel.HEADING_3,
      4: HeadingLevel.HEADING_4
    };

    const styleKey = level === 1 ? 'h1' : `h${level}`;
    const style = standards.headings[styleKey] || standards.headings.h4;

    return new Paragraph({
      heading: levelMap[level] || HeadingLevel.HEADING_4,
      children: [
        new TextRun({
          text: text,
          bold: style.bold,
          size: style.size * 2  // half-points
        })
      ],
      spacing: {
        before: style.spacing.before,
        after: style.spacing.after
      }
    });
  }

  /**
   * 단락 생성
   */
  createParagraph(text) {
    return new Paragraph({
      children: [
        new TextRun({
          text: text,
          size: standards.body.size * 2
        })
      ],
      spacing: {
        before: standards.body.spacing.before,
        after: standards.body.spacing.after
      }
    });
  }

  /**
   * 코드 블록 생성
   */
  createCodeBlock(code, language) {
    const lines = code.split('\n');
    const paragraphs = lines.map(line =>
      new Paragraph({
        children: [
          new TextRun({
            text: line || ' ',
            font: standards.fonts.code,
            size: standards.code.size * 2
          })
        ],
        spacing: { before: 0, after: 0 }
      })
    );

    return paragraphs;
  }

  /**
   * 표 생성 (간단 버전)
   */
  createTable(tokens) {
    // 간단한 구현 - 실제로는 더 복잡한 파싱 필요
    return new Paragraph({
      children: [
        new TextRun({
          text: '[표]',
          italics: true,
          color: '808080'
        })
      ]
    });
  }
}

module.exports = Converter;
