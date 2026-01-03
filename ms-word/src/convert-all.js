#!/usr/bin/env node
/**
 * 전체 챕터 일괄 변환 스크립트
 *
 * 사용법:
 *   node convert-all.js
 *   npm run convert:all
 */

const fs = require('fs-extra');
const path = require('path');
const chalk = require('chalk');
const Converter = require('./converter');

async function main() {
  const docsDir = path.join(__dirname, '../../docs');
  const outputDir = path.join(__dirname, '../output');

  console.log(chalk.blue('='.repeat(50)));
  console.log(chalk.blue('전체 챕터 일괄 변환'));
  console.log(chalk.blue('='.repeat(50)));
  console.log();

  // 출력 디렉토리 생성
  await fs.ensureDir(outputDir);

  // docs 폴더에서 챕터 파일 찾기
  const files = await fs.readdir(docsDir);
  const chapterFiles = files
    .filter(f => /^ch\d+\.md$/i.test(f))
    .sort();

  if (chapterFiles.length === 0) {
    console.log(chalk.yellow('변환할 챕터 파일이 없습니다.'));
    console.log(chalk.gray(`확인할 위치: ${docsDir}`));
    console.log(chalk.gray('파일 형식: ch01.md, ch02.md, ...'));
    return;
  }

  console.log(chalk.gray(`발견된 챕터: ${chapterFiles.length}개`));
  console.log();

  const converter = new Converter();
  let successCount = 0;
  let failCount = 0;

  for (const file of chapterFiles) {
    const inputPath = path.join(docsDir, file);
    const outputFile = file.replace('.md', '.docx');
    const outputPath = path.join(outputDir, outputFile);

    try {
      process.stdout.write(chalk.gray(`  ${file} → ${outputFile}... `));
      await converter.convert(inputPath, outputPath);
      console.log(chalk.green('✓'));
      successCount++;
    } catch (error) {
      console.log(chalk.red(`✗ ${error.message}`));
      failCount++;
    }
  }

  console.log();
  console.log(chalk.blue('='.repeat(50)));
  console.log(chalk.green(`✓ 성공: ${successCount}개`));
  if (failCount > 0) {
    console.log(chalk.red(`✗ 실패: ${failCount}개`));
  }
  console.log(chalk.blue('='.repeat(50)));
  console.log();
  console.log(chalk.gray(`출력 위치: ${outputDir}`));
}

main().catch(console.error);
