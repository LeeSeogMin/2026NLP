#!/usr/bin/env node
/**
 * 통합 도서 생성 스크립트
 *
 * 사용법:
 *   node create-book.js
 *   npm run create:book
 */

const fs = require('fs-extra');
const path = require('path');
const chalk = require('chalk');
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  PageBreak,
  AlignmentType
} = require('docx');

const metadata = require('../config/book-metadata.json');
const standards = require('../config/formatting-standards');
const Converter = require('./converter');

async function main() {
  const docsDir = path.join(__dirname, '../../docs');
  const outputDir = path.join(__dirname, '../output');
  const templatesDir = path.join(__dirname, '../templates');

  console.log(chalk.blue('='.repeat(50)));
  console.log(chalk.blue('통합 도서 생성'));
  console.log(chalk.blue('='.repeat(50)));
  console.log();
  console.log(chalk.gray(`제목: ${metadata.title}`));
  console.log(chalk.gray(`저자: ${metadata.author}`));
  console.log();

  await fs.ensureDir(outputDir);

  const sections = [];

  // 1. 표지 생성
  console.log(chalk.gray('  표지 생성...'));
  sections.push({
    properties: {},
    children: [
      new Paragraph({ text: '' }),
      new Paragraph({ text: '' }),
      new Paragraph({ text: '' }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({
            text: metadata.title,
            bold: true,
            size: 56
          })
        ]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({
            text: metadata.subtitle || '',
            size: 32
          })
        ],
        spacing: { before: 400 }
      }),
      new Paragraph({ text: '' }),
      new Paragraph({ text: '' }),
      new Paragraph({ text: '' }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({
            text: metadata.author,
            size: 28
          })
        ]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({
            text: `${metadata.year}`,
            size: 24
          })
        ],
        spacing: { before: 200 }
      })
    ]
  });

  // 2. 목차 (간단 버전)
  console.log(chalk.gray('  목차 생성...'));
  const tocChildren = [
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [
        new TextRun({ text: '목차', bold: true, size: 48 })
      ]
    }),
    new Paragraph({ text: '' })
  ];

  for (const chapter of metadata.chapters) {
    tocChildren.push(
      new Paragraph({
        children: [
          new TextRun({
            text: `제${chapter.number}장 ${chapter.title}`,
            size: 24
          })
        ],
        spacing: { before: 100, after: 100 }
      })
    );
  }

  sections.push({
    properties: {},
    children: tocChildren
  });

  // 3. 챕터 추가
  const converter = new Converter();

  for (const chapter of metadata.chapters) {
    const chapterPath = path.join(docsDir, chapter.file);

    if (await fs.pathExists(chapterPath)) {
      console.log(chalk.gray(`  제${chapter.number}장 추가...`));

      try {
        const markdown = await fs.readFile(chapterPath, 'utf-8');
        const tokens = converter.md.parse(markdown, {});
        const children = converter.tokensToDocx(tokens);

        sections.push({
          properties: {},
          children: [
            new Paragraph({
              children: [new PageBreak()]
            }),
            ...children
          ]
        });
      } catch (error) {
        console.log(chalk.yellow(`    경고: ${error.message}`));
      }
    } else {
      console.log(chalk.yellow(`  제${chapter.number}장 파일 없음: ${chapter.file}`));
    }
  }

  // 4. 문서 생성
  console.log(chalk.gray('  문서 생성 중...'));

  const doc = new Document({
    sections: sections
  });

  // 5. 저장
  const projectName = metadata.title
    .replace(/[^a-zA-Z0-9가-힣]/g, '-')
    .replace(/-+/g, '-')
    .toLowerCase();
  const outputPath = path.join(outputDir, `${projectName}-complete-book.docx`);

  const buffer = await Packer.toBuffer(doc);
  await fs.writeFile(outputPath, buffer);

  console.log();
  console.log(chalk.green('='.repeat(50)));
  console.log(chalk.green(`✓ 통합 도서 생성 완료`));
  console.log(chalk.green(`  ${outputPath}`));
  console.log(chalk.green('='.repeat(50)));
}

main().catch(error => {
  console.error(chalk.red('오류 발생:'), error.message);
  process.exit(1);
});
