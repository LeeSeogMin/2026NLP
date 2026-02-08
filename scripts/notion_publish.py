#!/usr/bin/env python3
"""
Markdown → Notion 변환 스크립트

docs/ch{N}.md 또는 contents.md를 Notion 페이지로 변환하여 발행한다.

사용법:
    python scripts/notion_publish.py --chapter 2     # 2장 발행
    python scripts/notion_publish.py --all            # 전체 발행
    python scripts/notion_publish.py --contents       # 목차 발행
    python scripts/notion_publish.py --chapter 2 --dry-run  # 파싱만 테스트

환경 변수 (.env):
    NOTION_API_KEY      - Notion Integration Token
    NOTION_DATABASE_ID  - 대상 Notion 데이터베이스 ID

Notion 설정 가이드:
    1. https://www.notion.so/my-integrations 에서 Integration 생성
    2. 대상 데이터베이스에 Integration 연결 (Share → Invite)
    3. .env 파일에 NOTION_API_KEY, NOTION_DATABASE_ID 설정
"""

import os
import re
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    from notion_client import Client
except ImportError:
    print("notion-client 패키지가 필요합니다: pip install notion-client")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================
# 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CONTENTS_FILE = PROJECT_ROOT / "contents.md"
PAGE_MAP_FILE = Path(__file__).resolve().parent / ".notion_pages.json"

MAX_BLOCKS_PER_REQUEST = 100
MAX_RICH_TEXT_LENGTH = 2000


# ============================================================
# 인라인 텍스트 파싱
# ============================================================

def parse_rich_text(text: str) -> List[Dict[str, Any]]:
    """마크다운 인라인 서식을 Notion rich_text 배열로 변환한다."""
    if not text or not text.strip():
        return [{"type": "text", "text": {"content": text or ""}}]

    result = []
    pattern = re.compile(
        r'(\[([^\]]+)\]\(([^)]+)\))'    # [text](url)
        r'|(\*\*`([^`]+)`\*\*)'          # **`bold code`**
        r'|(\*\*(.+?)\*\*)'              # **bold**
        r'|(\*(.+?)\*)'                   # *italic*
        r'|(`([^`]+)`)'                   # `code`
    )

    last_end = 0
    for match in pattern.finditer(text):
        if match.start() > last_end:
            plain = text[last_end:match.start()]
            if plain:
                result.append({"type": "text", "text": {"content": plain}})

        if match.group(1):      # 링크
            result.append({
                "type": "text",
                "text": {"content": match.group(2), "link": {"url": match.group(3)}}
            })
        elif match.group(4):    # 볼드+코드
            result.append({
                "type": "text",
                "text": {"content": match.group(5)},
                "annotations": {"bold": True, "code": True}
            })
        elif match.group(6):    # 볼드
            result.append({
                "type": "text",
                "text": {"content": match.group(7)},
                "annotations": {"bold": True}
            })
        elif match.group(8):    # 이탤릭
            result.append({
                "type": "text",
                "text": {"content": match.group(9)},
                "annotations": {"italic": True}
            })
        elif match.group(10):   # 인라인 코드
            result.append({
                "type": "text",
                "text": {"content": match.group(11)},
                "annotations": {"code": True}
            })

        last_end = match.end()

    if last_end < len(text):
        remaining = text[last_end:]
        if remaining:
            result.append({"type": "text", "text": {"content": remaining}})

    if not result:
        result.append({"type": "text", "text": {"content": text}})

    # 2000자 제한 처리
    truncated = []
    for item in result:
        content = item["text"]["content"]
        while len(content) > MAX_RICH_TEXT_LENGTH:
            chunk = _clone_rich_text(item, content[:MAX_RICH_TEXT_LENGTH])
            truncated.append(chunk)
            content = content[MAX_RICH_TEXT_LENGTH:]
        if content:
            truncated.append(_clone_rich_text(item, content))

    return truncated if truncated else result


def _clone_rich_text(item: Dict, content: str) -> Dict:
    """rich_text 항목을 복제하고 content를 교체한다."""
    clone = {"type": "text", "text": {"content": content}}
    if "link" in item.get("text", {}):
        clone["text"]["link"] = item["text"]["link"]
    if "annotations" in item:
        clone["annotations"] = dict(item["annotations"])
    return clone


# ============================================================
# 마크다운 블록 파싱
# ============================================================

LANG_MAP = {
    "python": "python", "py": "python",
    "bash": "bash", "sh": "bash", "shell": "bash",
    "javascript": "javascript", "js": "javascript",
    "json": "json", "markdown": "markdown", "md": "markdown",
    "html": "html", "css": "css", "sql": "sql",
    "yaml": "yaml", "yml": "yaml",
    "typescript": "typescript", "ts": "typescript",
    "tsx": "typescript", "jsx": "javascript",
    "mermaid": "plain text", "text": "plain text",
    "plain text": "plain text", "": "plain text",
}


def parse_markdown_to_blocks(filepath: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """마크다운 파일을 파싱하여 (제목, Notion 블록 리스트)를 반환한다."""
    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")
    blocks = []
    title = ""
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 빈 줄
        if not stripped:
            i += 1
            continue

        # 구분선
        if stripped == "---":
            blocks.append({"type": "divider", "divider": {}})
            i += 1
            continue

        # HTML 주석
        if stripped.startswith("<!--"):
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
            i += 1
            continue

        # 코드 블록
        if stripped.startswith("```"):
            language = stripped[3:].strip()
            notion_lang = LANG_MAP.get(language.lower(), "plain text")

            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # 닫는 ```

            code_content = "\n".join(code_lines)
            if len(code_content) > MAX_RICH_TEXT_LENGTH:
                code_content = code_content[:MAX_RICH_TEXT_LENGTH - 20] + "\n... (truncated)"

            blocks.append({
                "type": "code",
                "code": {
                    "rich_text": [{"type": "text", "text": {"content": code_content}}],
                    "language": notion_lang
                }
            })
            continue

        # 표
        if stripped.startswith("|"):
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row_line = lines[i].strip()
                if re.match(r'^\|[\s\-:]+\|$', row_line):
                    i += 1
                    continue
                cells = [c.strip() for c in row_line.split("|")[1:-1]]
                table_rows.append(cells)
                i += 1

            if table_rows:
                num_cols = max(len(row) for row in table_rows)
                for row in table_rows:
                    while len(row) < num_cols:
                        row.append("")

                children = []
                for row in table_rows:
                    children.append({
                        "type": "table_row",
                        "table_row": {
                            "cells": [parse_rich_text(cell) for cell in row]
                        }
                    })

                blocks.append({
                    "type": "table",
                    "table": {
                        "table_width": num_cols,
                        "has_column_header": True,
                        "has_row_header": False,
                        "children": children
                    }
                })
            continue

        # 제목
        heading_match = re.match(r'^(#{1,6})\s+(.+)', stripped)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2)

            if level == 1 and not title:
                title = heading_text
                i += 1
                continue

            notion_level = min(level, 3)
            heading_type = f"heading_{notion_level}"

            blocks.append({
                "type": heading_type,
                heading_type: {
                    "rich_text": parse_rich_text(heading_text),
                    "is_toggleable": False
                }
            })
            i += 1
            continue

        # 블록 인용
        if stripped.startswith(">"):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_text = lines[i].strip().lstrip(">").strip()
                quote_lines.append(quote_text)
                i += 1

            quote_content = " ".join(quote_lines)

            # 특별 인용은 callout으로
            callout_map = {
                "미션": ("🎯", "blue_background"),
                "Copilot 활용": ("🤖", "gray_background"),
                "라이브 코딩": ("💻", "yellow_background"),
                "강의 팁": ("💡", "green_background"),
            }

            used_callout = False
            for keyword, (emoji, color) in callout_map.items():
                if keyword in quote_content:
                    blocks.append({
                        "type": "callout",
                        "callout": {
                            "rich_text": parse_rich_text(quote_content),
                            "icon": {"type": "emoji", "emoji": emoji},
                            "color": color
                        }
                    })
                    used_callout = True
                    break

            if not used_callout:
                blocks.append({
                    "type": "quote",
                    "quote": {"rich_text": parse_rich_text(quote_content)}
                })
            continue

        # 순서 없는 목록
        if re.match(r'^[-*]\s+', stripped):
            list_text = re.sub(r'^[-*]\s+', '', stripped)
            block = {
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": parse_rich_text(list_text)
                }
            }
            blocks.append(block)
            i += 1

            # 하위 목록
            while i < len(lines):
                indent_match = re.match(r'^(\s{2,})[-*]\s+(.+)', lines[i])
                if indent_match:
                    sub_text = indent_match.group(2)
                    if "children" not in block["bulleted_list_item"]:
                        block["bulleted_list_item"]["children"] = []
                    block["bulleted_list_item"]["children"].append({
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": parse_rich_text(sub_text)
                        }
                    })
                    i += 1
                else:
                    break
            continue

        # 순서 있는 목록
        if re.match(r'^\d+\.\s+', stripped):
            list_text = re.sub(r'^\d+\.\s+', '', stripped)
            blocks.append({
                "type": "numbered_list_item",
                "numbered_list_item": {
                    "rich_text": parse_rich_text(list_text)
                }
            })
            i += 1
            continue

        # 일반 단락
        para_lines = [stripped]
        i += 1
        stop_patterns = ("#", "|", ">", "```", "---", "<!--")
        while i < len(lines):
            next_stripped = lines[i].strip()
            if not next_stripped:
                break
            if any(next_stripped.startswith(p) for p in stop_patterns):
                break
            if re.match(r'^[-*]\s+', next_stripped):
                break
            if re.match(r'^\d+\.\s+', next_stripped):
                break
            para_lines.append(next_stripped)
            i += 1

        para_text = " ".join(para_lines)
        blocks.append({
            "type": "paragraph",
            "paragraph": {"rich_text": parse_rich_text(para_text)}
        })

    return title, blocks


# ============================================================
# Notion API
# ============================================================

def load_page_map() -> Dict[str, str]:
    """저장된 페이지 ID 매핑을 로드한다."""
    if PAGE_MAP_FILE.exists():
        return json.loads(PAGE_MAP_FILE.read_text(encoding="utf-8"))
    return {}


def save_page_map(page_map: Dict[str, str]):
    """페이지 ID 매핑을 저장한다."""
    PAGE_MAP_FILE.write_text(
        json.dumps(page_map, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def clear_page_content(notion: Client, page_id: str):
    """기존 페이지의 모든 블록을 삭제한다."""
    while True:
        children = notion.blocks.children.list(block_id=page_id, page_size=100)
        results = children.get("results", [])
        if not results:
            break
        for block in results:
            try:
                notion.blocks.delete(block_id=block["id"])
            except Exception:
                pass


def append_blocks(notion: Client, page_id: str, blocks: List[Dict]):
    """블록을 배치로 나누어 추가한다."""
    for i in range(0, len(blocks), MAX_BLOCKS_PER_REQUEST):
        batch = blocks[i:i + MAX_BLOCKS_PER_REQUEST]
        try:
            notion.blocks.children.append(block_id=page_id, children=batch)
        except Exception as e:
            print(f"  블록 추가 실패 ({i}~{i + len(batch)}): {e}")
            for j, block in enumerate(batch):
                try:
                    notion.blocks.children.append(
                        block_id=page_id, children=[block]
                    )
                except Exception as e2:
                    print(f"    블록 {i + j} 건너뜀: {e2}")


def publish_page(
    notion: Client,
    database_id: str,
    title: str,
    blocks: List[Dict],
    chapter_num: Optional[int] = None,
    map_key: str = "",
    page_map: Optional[Dict[str, str]] = None,
) -> str:
    """Notion 데이터베이스에 페이지를 생성하거나 업데이트한다."""
    existing_id = (page_map or {}).get(map_key)

    if existing_id:
        print(f"  기존 페이지 업데이트: {map_key}")
        try:
            notion.pages.update(
                page_id=existing_id,
                properties={
                    "title": {"title": [{"text": {"content": title}}]},
                },
            )
            clear_page_content(notion, existing_id)
            append_blocks(notion, existing_id, blocks)
            return existing_id
        except Exception as e:
            print(f"  업데이트 실패, 새 페이지 생성: {e}")

    print(f"  새 페이지 생성: {title}")
    properties: Dict[str, Any] = {
        "title": {"title": [{"text": {"content": title}}]},
    }
    if chapter_num is not None:
        properties["Chapter"] = {"number": chapter_num}

    new_page = notion.pages.create(
        parent={"database_id": database_id},
        properties=properties,
        children=blocks[:MAX_BLOCKS_PER_REQUEST],
    )
    page_id = new_page["id"]

    if len(blocks) > MAX_BLOCKS_PER_REQUEST:
        append_blocks(notion, page_id, blocks[MAX_BLOCKS_PER_REQUEST:])

    return page_id


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Markdown → Notion 변환 스크립트"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chapter", type=int, help="발행할 장 번호")
    group.add_argument("--all", action="store_true", help="전체 발행")
    group.add_argument("--contents", action="store_true", help="목차 발행")
    parser.add_argument(
        "--dry-run", action="store_true", help="API 호출 없이 파싱만 테스트"
    )
    args = parser.parse_args()

    api_key = os.environ.get("NOTION_API_KEY")
    database_id = os.environ.get("NOTION_DATABASE_ID")

    if not args.dry_run:
        if not api_key:
            print("NOTION_API_KEY 환경 변수를 설정하세요.")
            print("  .env 파일에 NOTION_API_KEY=secret_xxx 추가")
            sys.exit(1)
        if not database_id:
            print("NOTION_DATABASE_ID 환경 변수를 설정하세요.")
            print("  .env 파일에 NOTION_DATABASE_ID=xxx 추가")
            sys.exit(1)
        notion = Client(auth=api_key)
    else:
        notion = None

    page_map = load_page_map()

    # 발행 대상 결정
    targets: List[Tuple[str, Path, Optional[int]]] = []

    if args.contents:
        if not CONTENTS_FILE.exists():
            print(f"파일 없음: {CONTENTS_FILE}")
            sys.exit(1)
        targets.append(("contents", CONTENTS_FILE, None))

    elif args.chapter:
        ch_file = DOCS_DIR / f"ch{args.chapter}.md"
        if not ch_file.exists():
            print(f"파일 없음: {ch_file}")
            sys.exit(1)
        targets.append((f"ch{args.chapter}", ch_file, args.chapter))

    elif args.all:
        if CONTENTS_FILE.exists():
            targets.append(("contents", CONTENTS_FILE, None))
        for ch_file in sorted(DOCS_DIR.glob("ch*.md")):
            m = re.match(r"ch(\d+)\.md", ch_file.name)
            if m:
                targets.append((f"ch{m.group(1)}", ch_file, int(m.group(1))))

    for map_key, filepath, ch_num in targets:
        print(f"\n{filepath.name} 처리 중...")

        title, blocks = parse_markdown_to_blocks(filepath)
        print(f"  제목: {title}")
        print(f"  블록: {len(blocks)}개")

        if args.dry_run:
            counts: Dict[str, int] = {}
            for b in blocks:
                t = b["type"]
                counts[t] = counts.get(t, 0) + 1
            for t, c in sorted(counts.items()):
                print(f"    {t}: {c}")
            continue

        try:
            page_id = publish_page(
                notion=notion,
                database_id=database_id,
                title=title,
                blocks=blocks,
                chapter_num=ch_num,
                map_key=map_key,
                page_map=page_map,
            )
            page_map[map_key] = page_id
            save_page_map(page_map)
            clean_id = page_id.replace("-", "")
            print(f"  발행 완료: https://notion.so/{clean_id}")
        except Exception as e:
            print(f"  발행 실패: {e}")

    print("\n완료!")


if __name__ == "__main__":
    main()
