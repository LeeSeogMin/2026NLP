#!/usr/bin/env node
/**
 * 개별 챕터 변환 스크립트
 *
 * 사용법:
 *   node convert-chapter.js <chapter-number>
 *   npm run convert:chapter 2
 */

const fs = require('fs-extra');
const path = require('path');
const chalk = require('chalk');
const Converter = require('./converter');

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log(chalk.red('오류: 챕터 번호를 지정하세요.'));
    console.log(chalk.yellow('사용법: npm run convert:chapter <number>'));
    console.log(chalk.yellow('예시: npm run convert:chapter 2'));
    process.exit(1);
  }

  const chapterNum = parseInt(args[0]);
  if (isNaN(chapterNum)) {
    console.log(chalk.red('오류: 유효한 챕터 번호를 입력하세요.'));
    process.exit(1);
  }

  // 파일 경로 설정
  const docsDir = path.join(__dirname, '../../docs');
  const outputDir = path.join(__dirname, '../output');

  // 챕터 파일 찾기
  const paddedNum = chapterNum.toString().padStart(2, '0');
  const possibleNames = [
    `ch${paddedNum}.md`,
    `ch${chapterNum}.md`,
    `chapter${paddedNum}.md`,
    `chapter${chapterNum}.md`
  ];

  let inputPath = null;
  for (const name of possibleNames) {
    const testPath = path.join(docsDir, name);
    if (await fs.pathExists(testPath)) {
      inputPath = testPath;
      break;
    }
  }

  if (!inputPath) {
    console.log(chalk.red(`오류: 챕터 ${chapterNum} 파일을 찾을 수 없습니다.`));
    console.log(chalk.yellow(`확인할 위치: ${docsDir}`));
    process.exit(1);
  }

  // 출력 디렉토리 생성
  await fs.ensureDir(outputDir);

  const outputPath = path.join(outputDir, `ch${paddedNum}.docx`);

  // 변환 실행
  console.log(chalk.blue('='.repeat(50)));
  console.log(chalk.blue(`챕터 ${chapterNum} 변환 시작`));
  console.log(chalk.blue('='.repeat(50)));
  console.log();
  console.log(chalk.gray(`입력: ${inputPath}`));
  console.log(chalk.gray(`출력: ${outputPath}`));
  console.log();

  try {
    const converter = new Converter();
    await converter.convert(inputPath, outputPath);

    console.log();
    console.log(chalk.green('='.repeat(50)));
    console.log(chalk.green(`✓ 변환 완료: ${outputPath}`));
    console.log(chalk.green('='.repeat(50)));

  } catch (error) {
    console.log(chalk.red(`✗ 변환 실패: ${error.message}`));
    console.error(error);
    process.exit(1);
  }
}

main();
