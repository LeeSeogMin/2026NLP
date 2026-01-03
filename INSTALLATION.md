# 설치 가이드

이 문서는 연구 프로젝트에 필요한 모든 도구와 라이브러리 설치 방법을 안내합니다.

## 시스템 요구사항

- **OS**: macOS, Linux, Windows
- **Python**: 3.9 이상 (현재: Python 3.13.7)
- **저장공간**: 최소 5GB 여유 공간

## 설치 단계

### 1. Pandoc 설치

#### macOS (Homebrew)
```bash
brew install pandoc
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install pandoc
```

#### Windows (Chocolatey)
```bash
choco install pandoc
```

설치 확인:
```bash
pandoc --version
# 출력: pandoc 3.8.3 (또는 그 이상)
```

### 2. Python 가상환경 생성

프로젝트 루트 디렉토리에서:

```bash
cd /Users/callii/Documents/research_project

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows
```

### 3. Python 라이브러리 설치

가상환경이 활성화된 상태에서:

```bash
pip install -r requirements.txt
```

**설치되는 주요 패키지**:
- `pypandoc`: Pandoc Python 래퍼
- `python-docx`: Word 문서 처리
- `pandas`, `numpy`: 데이터 처리
- `scikit-learn`: 머신러닝
- `matplotlib`, `seaborn`: 시각화
- `scipy`: 과학 계산
- `tqdm`: 진행률 표시
- `pytest`: 테스트

### 4. 설치 검증

```bash
python test_installation.py
```

**예상 출력**:
```
============================================================
연구 프로젝트 도구 설치 검증
============================================================

✅ Python: 3.13.7

📦 외부 도구:
✅ Pandoc: pandoc 3.8.3

🐍 Python 패키지:
✅ pypandoc: 1.16.2
✅ python-docx: 1.2.0
✅ pandas: 2.3.3
✅ numpy: 2.4.0
...

✅ 모든 도구가 올바르게 설치되었습니다!
```

## 추가 도구 (선택사항)

### Jupyter Notebook

데이터 분석 및 실험 시 유용합니다.

```bash
pip install jupyter ipykernel

# Jupyter Notebook 실행
jupyter notebook
```

### Deep Learning 프레임워크

딥러닝 연구 시 필요합니다.

#### PyTorch
```bash
pip install torch torchvision
```

#### TensorFlow
```bash
pip install tensorflow
```

### MLflow (실험 추적)

```bash
pip install mlflow
```

## 사용 방법

### 가상환경 활성화 (매번 필요)

프로젝트 작업 시 항상 가상환경을 활성화하세요:

```bash
cd /Users/callii/Documents/research_project
source venv/bin/activate
```

활성화 확인:
```bash
which python
# 출력: /Users/callii/Documents/research_project/venv/bin/python
```

### 가상환경 비활성화

```bash
deactivate
```

## 문제 해결

### Pandoc 설치 실패

**문제**: `brew install pandoc` 실패

**해결**:
1. Homebrew 업데이트: `brew update`
2. 다시 시도: `brew install pandoc`
3. 또는 공식 웹사이트에서 직접 다운로드: https://pandoc.org/installing.html

### Python 패키지 설치 실패

**문제**: `externally-managed-environment` 오류

**해결**: 가상환경을 사용하세요 (위 단계 2 참조)

**문제**: 특정 패키지 설치 실패

**해결**:
```bash
pip install --upgrade pip
pip install <package-name> --no-cache-dir
```

### NumPy/SciPy 설치 오류

**문제**: Apple Silicon (M1/M2) Mac에서 빌드 실패

**해결**:
```bash
# Rosetta를 통해 설치
arch -x86_64 pip install numpy scipy
```

## 업데이트

라이브러리 업데이트:

```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

## 검증 체크리스트

설치 후 다음을 확인하세요:

- [ ] Pandoc 설치 완료 (`pandoc --version`)
- [ ] Python 가상환경 생성 완료 (`venv/` 폴더 존재)
- [ ] 모든 Python 패키지 설치 완료 (`python test_installation.py`)
- [ ] 변환 스크립트 실행 가능 (`python conversion/md_to_docx.py --help`)

## 다음 단계

설치 완료 후:

1. [README.md](README.md) 읽기 - 프로젝트 개요 이해
2. [tempdocs/object.md](tempdocs/object.md) 읽기 - 워크플로우 상세
3. Phase 1 시작 - 연구 주제 탐색

---

**문제가 계속되면**:
- GitHub Issues에 보고
- 또는 프로젝트 관리자에게 문의
