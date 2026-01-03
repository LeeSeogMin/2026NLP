# MCP 서버 사용 가이드

## 개요

이 프로젝트는 Model Context Protocol (MCP) 서버를 통해 SearchRAG 도구를 Claude Code와 통합했습니다.

## MCP 서버 구성

### 1. SearchRAG MCP 서버
**위치**: `/Users/callii/Documents/SearchRAG/mcp_server.py`

**제공 도구**:
1. `search_papers` - 학술 논문 검색
2. `analyze_research_gap` - 연구 갭 분석
3. `extract_citations` - 인용 추출 및 검증
4. `generate_search_queries` - 검색 쿼리 생성

### 2. Filesystem MCP 서버
**범위**: `/Users/callii/Documents/research_project 2`

**기능**: 프로젝트 디렉토리 내 파일 시스템 접근

## MCP 서버 시작 방법

### Claude Code에서 자동 시작

```bash
# mcp.json 설정이 있는 프로젝트에서 Claude Code 실행
claude
```

Claude Code는 자동으로 `mcp.json`을 감지하고 MCP 서버를 시작합니다.

### 수동 테스트

```bash
# MCP 서버 테스트
cd /Users/callii/Documents/research_project\ 2
python3 test_mcp_server.py
```

## 사용 예시

### 1. 논문 검색

Claude Code에서:
```
"SearchRAG MCP 서버를 사용해서 'big data anomaly detection' 관련 논문을 20개 검색해줘"
```

MCP 서버가 자동으로 호출됩니다:
```json
{
  "tool": "search_papers",
  "arguments": {
    "query": "big data anomaly detection",
    "limit": 20,
    "year_from": 2020
  }
}
```

### 2. 검색 쿼리 생성

```
"연구 주제 '머신러닝 기반 이상 탐지'로부터 검색 쿼리를 3개 생성해줘"
```

MCP 서버 응답:
```json
{
  "success": true,
  "queries": [
    "머신러닝 기반 이상 탐지",
    "머신러닝 기반 이상 탐지 machine learning",
    "머신러닝 기반 이상 탐지 deep learning"
  ]
}
```

### 3. 인용 추출

```
"다음 텍스트에서 인용을 추출해줘:
Smith et al. (2020) demonstrated that deep learning models..."
```

MCP 서버가 인용 패턴을 분석하고 추출합니다.

## Phase별 MCP 활용

### Phase 1: 연구 주제 탐색
- `generate_search_queries`: 연구 주제로부터 검색 쿼리 생성

### Phase 2: 문헌 검토
- `search_papers`: 학술 논문 검색
- `analyze_research_gap`: 연구 갭 분석
- `extract_citations`: 인용 추출

### Phase 6: 논문 초안 작성
- `extract_citations`: 논문 내 인용 검증

## 설정 파일

### mcp.json

```json
{
  "mcpServers": {
    "searchrag": {
      "command": "python3",
      "args": [
        "/Users/callii/Documents/SearchRAG/mcp_server.py"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "PYTHONPATH": "/Users/callii/Documents/SearchRAG"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/callii/Documents/research_project 2"
      ]
    }
  }
}
```

## 문제 해결

### MCP 서버가 시작되지 않을 때

1. **Python 경로 확인**
   ```bash
   which python3
   ```

2. **SearchRAG 모듈 확인**
   ```bash
   cd /Users/callii/Documents/SearchRAG
   python3 -c "from search.semantic_scholar import SemanticScholarAPI; print('OK')"
   ```

3. **환경 변수 확인**
   ```bash
   echo $ANTHROPIC_API_KEY
   ```

### MCP 서버 로그 확인

Claude Code 실행 시 MCP 서버 오류는 터미널에 출력됩니다.

```bash
# 디버그 모드로 실행
claude --verbose
```

### 수동으로 MCP 서버 테스트

```bash
cd /Users/callii/Documents/research_project\ 2
python3 test_mcp_server.py
```

모든 테스트가 통과해야 합니다:
- [Test 1] Initialize Request ✓
- [Test 2] List Tools Request ✓
- [Test 3] Generate Search Queries ✓

## MCP 프로토콜 버전

- **프로토콜 버전**: 2024-11-05
- **서버 이름**: searchrag
- **서버 버전**: 1.0.0

## 보안 고려사항

1. **API 키 관리**
   - API 키는 `.env` 파일에만 저장
   - `mcp.json`에서 환경 변수로 참조 (`${ANTHROPIC_API_KEY}`)
   - Git에 커밋하지 않음 (`.gitignore`에 포함)

2. **파일 시스템 접근 제한**
   - Filesystem MCP 서버는 프로젝트 디렉토리만 접근
   - 상위 디렉토리 접근 불가

3. **권한 설정**
   - `.claude/settings.local.json`에서 허용된 명령어만 실행

## 참고 자료

- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [Claude Code MCP 가이드](https://docs.anthropic.com/claude-code/mcp)
- SearchRAG 문서: [tempdocs/searchrag.md](tempdocs/searchrag.md)

---

**작성일**: 2025-12-31
**버전**: 1.0
**Agentic Layer 등급**: Class 1 등급 3 달성
