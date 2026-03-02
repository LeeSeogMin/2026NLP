# Git & GitHub 빠른 시작 가이드

## 1단계: Git 설치 (3분)

**Windows**: https://git-scm.com/download/win → 다운로드 후 설치 (기본값 OK)

**macOS**: 터미널에서 실행

```bash
xcode-select --install
```

설치 확인:

```bash
git --version
```

## 2단계: GitHub 가입 (2분)

1. https://github.com 접속
2. "Sign up" → 학교 이메일로 가입
3. 이메일 인증

## 3단계: 로컬 연결 (1분)

> VSCode 통합 터미널(Ctrl+`), CMD, PowerShell 중 어디서든 실행 가능. VSCode 통합 터미널 권장.

```bash
git config --global user.name "홍길동"
git config --global user.email "학교이메일@ac.kr"
```


# GitHub Copilot 대학생 무료 사용 방법

GitHub는 **Copilot Free** (제한된 무료 플랜)와 **Copilot Pro** (고급 기능 풀버전)을 구분해서 운영하고 있으며,

**검증된 학생**은 **Copilot Pro**를 **학생 신분 유지 기간 동안 완전 무료**로 사용할 수 있습니다.

### 현재(2026년) 상황 요약

| 구분               | 대상자                     | 가격     | 주요 제한사항                                   | 모델/기능 수준          |
| ------------------ | -------------------------- | -------- | ----------------------------------------------- | ----------------------- |
| Copilot Free       | 누구나                     | 무료     | 월 2,000 코드 완성, 50 채팅 등 매우 제한적      | 기본 모델               |
| Copilot Pro        | 일반인                     | 월 $10   | 300 premium requests + 추가 과금 가능           | 최신 모델 풀 액세스     |
| Copilot Pro (학생) | GitHub Education 검증 학생 | **무료** | 월 300 premium requests (초과 시 다음달 초기화) | Pro 풀 기능 + 최신 모델 |

→ **대학생이 진짜 원하는 건 Copilot Pro 무료**이며, 이를 위해서는 **GitHub Student Developer Pack** 승인이 필수입니다.

### 2026년 최신 정확한 등록 절차 (단계별)

1. **GitHub 계정 준비**
   - 이미 계정이 있다면 로그인
   - 없다면 [https://github.com](https://github.com/) 에서 새로 생성 (학교 이메일 추천)
   - 기존 계정이 있으면 설정(https://github.com/settings/emails)에서 학교 이메일(.ac.kr)을 추가
   - **학교 이메일을 primary email(기본 이메일)로 설정** (드롭다운에서 선택 후 Save → 인증 인식을 도움. 나중에 개인 이메일로 되돌릴 수 있음)
   - 이메일 추가 후 verification link(인증 링크)를 클릭해 verified 상태로 만들기
2. **GitHub Student Developer Pack 신청 페이지 이동**

   https://education.github.com/pack

   또는 [https://education.github.com](https://education.github.com/) → "Get your pack" 클릭

3. **학생 신분 증명** (가장 중요한 단계)

   대부분의 한국 대학생이 성공하는 순서 (우선순위 높은 순) :

   | 순위 | 증빙 방법                       | 성공률    | 소요시간    | 비고                                        |
   | ---- | ------------------------------- | --------- | ----------- | ------------------------------------------- |
   | 1    | 학교 공식 이메일 (.ac.kr)       | 매우 높음 | 즉시~수시간 | 대부분 자동 승인                            |
   | 2    | 학생증 사진 (재학증명서) 업로드 | 높음      | 1~5일       | 선명하게 촬영, 이름·학번·유효기간 보여야 함 |
   | 3    | 재학증명서 pdf 업로드           | 높음      | 1~7일       | 최근 3개월 이내 발급본                      |
   | 4    | 등록금 영수증 + 신분증          | 중간      | 3~10일      | 최후의 수단                                 |

   → **한국 4년제 대학 재학생이라면 대부분 학교 이메일만으로 1~24시간 내 자동 승인**됩니다.

4. **승인 확인**
   - https://education.github.com/pack 에서 "Your pack" 상태 확인
   - 승인 메일 도착 (보통 "You're all set!" 제목)
5. **Copilot Pro 무료 활성화** (승인 후 바로 가능)

   두 가지 방법 중 편한 것 선택:

   방법 A (가장 확실)
   - https://github.com/settings/copilot 이동
   - "Code, planning, and automation" → Copilot 클릭
   - 학생 혜택으로 무료 가입 버튼 나타남 → 클릭

   방법 B
   - https://github.com/features/copilot 로 이동
   - 학생으로 인식되면 "무료로 시작" 또는 "Claim free access" 버튼 등장

   방법 C (학생/교사 전용 무료 signup 페이지)
   - https://github.com/github-copilot/free_signup 으로 직접 이동

   > **주의**: 신용카드 입력이 요구되면 진행하지 마십시오. 학생 혜택은 완전 무료이며 결제 정보가 필요하지 않습니다.

6. **VS Code 등 에디터에서 사용 시작**
   - GitHub 계정으로 Copilot 확장 로그인
   - 학생 혜택이 정상 적용되어 풀 Pro 기능 사용 가능

### 주의사항 (2026년 기준 자주 발생하는 문제)

- 승인 후에도 바로 안 보일 때 → 72시간까지 기다린 뒤 재로그인 시도 (Incognito/시크릿 모드 사용 추천). 혜택 동기화에 72시간~최대 2주가 소요될 수 있음
- "무료 버튼이 안 보임" → 캐시 지우기 / 다른 브라우저 시도 / primary email을 학교 이메일로 재설정 후 대기 / https://github.com/settings/copilot 직접 들어가기
- 인증 실패 시 → GitHub Support(https://support.github.com/contact/education)에 티켓 제출 (카테고리: "Student having trouble redeeming offers")
- 월 300 premium requests 제한은 학생도 동일 (과거에는 무제한이었으나 2025년 중반부터 변경됨)
- 졸업하면 자동으로 Pro 유료 전환됨 → 재학생 기간에 최대한 활용 권장
- 공식 문서 참조: https://docs.github.com/en/education

위 방법은 2026년 2월 6일 기준 GitHub 공식 문서 및 실제 학생 사례들을 종합한 **현재 가장 정확한 절차**입니다.

학교 이메일이 있다면 거의 100% 성공한다고 봐도 무방합니다.

# GitHub Copilot 사용 설명서

이 문서는 VS Code에서 GitHub Copilot을 활용하는 방법을 안내합니다. 아래 캡처 이미지를 참고하여 주요 기능과 메뉴를 설명합니다.

---

## 1. Copilot Chat 세션 시작

- **새 챗 세션(New Chat Session)**: Ctrl+N 단축키로 새로운 Copilot Chat 세션을 시작할 수 있습니다.
  **Continue In**: 챗 세션을 실행할 환경을 선택할 수 있습니다.
  - Local(로컬): 현재 PC에서 직접 Copilot 챗을 실행합니다. 빠른 응답과 파일 접근이 가능합니다.
  - Background(@cli): 명령줄 환경에서 Copilot을 실행합니다. 백그라운드 작업이나 자동화에 적합합니다.
  - Cloud(@cloud): 클라우드 서버에서 Copilot 챗을 실행합니다. 대용량 작업, 서버 리소스 활용, 원격 협업에 유리합니다.

---

## 2. 에이전트 및 역할 선택

- **Agent 메뉴**: Ctrl+Shift+I로 Copilot의 역할을 선택할 수 있습니다.
  - coder: 코드 작성
  - graphic: 그래픽/다이어그램 생성
  - planner: 집필 계획
  - researcher: 자료 조사
  - reviewer: 품질 검토
  - writer: 원고 작성
- **Plan/Ask**: 계획 수립 또는 질문 모드로 전환 가능
- **커스텀 에이전트 구성**: 필요에 따라 직접 에이전트 역할을 추가/설정할 수 있습니다.

---

## 3. 도구(툴) 설정

- **Configure Tools**: Copilot이 사용할 수 있는 도구를 선택/해제할 수 있습니다.
  - 예시: edit(파일 편집), execute(코드 실행), search(검색), todo(할 일 관리), web(웹 정보 수집) 등
  - MCP 서버 기반의 확장 도구도 활성화 가능
- **도구 선택 후 OK 버튼으로 적용**

---

## 4. MCP 서버 및 확장 기능

- **MCP 서버 알림**: 새로운 MCP 서버(예: mcp-server-time, GitHub 등)가 활성화되면 알림이 표시됩니다.
- **자동 시작 옵션**: "Automatically start MCP servers" 체크박스로 서버 자동 실행 설정 가능

---

## 5. Copilot 모델 선택

- **모델 선택**: 하단 메뉴에서 GPT-4.1 등 Copilot이 사용할 모델을 선택할 수 있습니다.

---

## 6. 기타 기능

- **챗 세션 관리**: 여러 챗 세션을 동시에 운영 가능
- **도구/에이전트/모델 조합**: 작업 목적에 따라 자유롭게 조합하여 활용

---

## 참고

- Copilot은 코드 작성, 문서화, 그래픽 생성, 품질 검토 등 다양한 역할을 수행할 수 있습니다.
- 각 메뉴와 도구는 프로젝트 목적에 맞게 선택하여 사용하세요.

---

# 실습 환경 자동 설정 (setup_env.py)

15주차 전체 실습에 필요한 Python 환경을 **한 번에** 구축하는 스크립트이다. GPU를 자동 감지하여 최적의 PyTorch를 설치하고, 모든 실습 패키지를 일괄 설치한다.

## 사전 요구사항

- **Python 3.10 이상** (https://www.python.org/downloads/)
- **(선택) NVIDIA GPU**: 드라이버가 설치되어 있으면 CUDA 버전을 자동 감지한다

Python 설치 확인:

```bash
python --version
```

## 실행 방법

> VSCode 통합 터미널(Ctrl+`) 또는 CMD/PowerShell에서 실행한다. 프로젝트 루트(`2026NLP/`) 디렉토리에서 실행해야 한다.

### 기본 실행 (자동 감지)

```bash
python scripts/setup_env.py
```

GPU가 있으면 자동으로 CUDA/MPS 버전 PyTorch를 설치하고, 없으면 CPU 버전을 설치한다.

### 옵션

```bash
python scripts/setup_env.py --cpu         # GPU 무시, CPU 버전 강제 설치
python scripts/setup_env.py --cuda 12.1   # CUDA 버전 수동 지정
python scripts/setup_env.py --skip-venv   # 이미 가상환경 안에 있을 때
```

## 스크립트가 수행하는 작업 (7단계)

| 단계 | 내용 | 소요시간 |
|------|------|----------|
| 1단계 | 시스템 정보 확인 (OS, Python 버전) | 즉시 |
| 2단계 | GPU 자동 감지 (NVIDIA CUDA / Apple MPS / CPU) | 즉시 |
| 3단계 | `venv` 가상환경 생성 + pip 업그레이드 | ~30초 |
| 4단계 | GPU에 맞는 PyTorch 설치 | 2~5분 |
| 5단계 | `requirements.txt` 전체 패키지 설치 | 5~10분 |
| 6단계 | 설치 검증 (핵심 패키지 13개 + GPU 설정) | ~10초 |
| 7단계 | GPU 벤치마크 (4096×4096 행렬 곱 CPU vs GPU) | ~30초 |

## 설치 후 사용

가상환경 활성화:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

실습 코드 실행:

```bash
python practice/chapter1/code/1-3-텐서기초.py
```

## 지원 환경

| OS | GPU | PyTorch 설치 방식 |
|----|-----|-------------------|
| Windows 10/11 | NVIDIA (CUDA 11.6~12.6) | CUDA 자동 감지 설치 |
| macOS (Apple Silicon) | M1/M2/M3/M4 | MPS 가속 자동 적용 |
| macOS (Intel) / Linux | 없음 | CPU 버전 설치 |

## 문제 해결

- **"Python 3.10 이상이 필요합니다"**: Python 최신 버전을 설치한다 (https://www.python.org/downloads/)
- **PyTorch 설치 실패**: `--cpu` 옵션으로 CPU 버전을 먼저 설치한 뒤, 나중에 GPU 버전으로 재설치한다
- **일부 패키지 설치 실패**: 선택 패키지(bitsandbytes 등)는 실패해도 대부분의 실습에 영향 없다. 해당 주차에서 필요 시 개별 설치한다
- **GPU가 감지되지 않음**: NVIDIA 드라이버가 설치되어 있는지 확인한다 (`nvidia-smi` 명령 실행). 드라이버 설치 후 스크립트를 다시 실행하면 된다

---
