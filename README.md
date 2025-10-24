# AI 제품 추천 챗봇

OpenAI GPT 모델을 사용한 제품 추천 챗봇 애플리케이션입니다.
인텔 최종 프로젝트에 사용했던 코드 입니다. 
<<24.01.25 ~ 24.02.07>>

> local에서 진행을 했던 프로젝트를 보완하여 파이썬 프로젝트로 올립니다.

(기존에 사용하였던 PyPDF2가 버전이 높아지면서 API 가 바뀌어 해당 부분을 수정하고 , 실행파일, 설치파일, 주석을 추가하여 업로드하였습니다.)

## 기능

- PyQt5 기반 GUI 채팅 인터페이스
- 대화 히스토리 관리
- PDF 문서 업로드 및 검색 기능
- OpenAI API 연동
- Fine-tuning 지원

## 프로젝트 구조

```
chatboot/
├── src/                 # 소스 코드
│   ├── main.py          # 기본 챗봇
│   ├── main_with_pdf.py # PDF 기능 포함 챗봇
│   ├── main_advanced.py # 고급 UI 버전
│   ├── pdf_processor.py # PDF 처리 모듈
│   └── fine_tuning.py   # Fine-tuning 도구
├── scripts/             # 실행 스크립트
│   ├── run.py           # 자동 검증 실행 스크립트
│   └── install.sh       # 설치 스크립트
├── data/                # 데이터 파일 (임베딩, 학습 데이터 등)
├── app.py               # 메인 실행 파일 (간편 실행)
├── config.py            # 설정 파일
├── requirements.txt     # 패키지 목록
├── .env.example         # 환경변수 예시
└── README.md           # 프로젝트 설명
```

## 설치

필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

또는 자동 설치 스크립트 사용:

```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

## 설정

OpenAI API 키를 설정합니다. 다음 중 하나의 방법을 선택:

### 방법 1: .env 파일 사용
```bash
cp .env.example .env
```
그 후 .env 파일을 편집하여 실제 API 키를 입력합니다.

### 방법 2: config.py 직접 수정
```python
OPENAI_API_KEY = "sk-your-api-key-here"
```

## 실행

### 간편 실행 
```bash
python app.py
```

대화형 메뉴에서 원하는 버전을 선택하거나:

```bash
python app.py basic      # 기본 챗봇
python app.py pdf        # PDF 기능 포함
python app.py advanced   # 고급 UI
```

### 직접 실행
```bash
# 기본 챗봇
python src/main.py

# PDF 기능 포함 챗봇
python src/main_with_pdf.py

# 고급 UI 챗봇
python src/main_advanced.py
```

### 검증 스크립트 사용
```bash
python scripts/run.py
```

## PDF 기능 사용법

1. PDF 기능 포함 챗봇을 실행합니다
2. "PDF 로드" 버튼을 클릭하여 PDF 파일을 선택합니다
3. PDF 처리가 완료되면 문서 내용에 대해 질문할 수 있습니다

## Fine-tuning

특정 제품 데이터로 모델을 커스터마이징할 수 있습니다:

```python
import sys
sys.path.append('src')
from fine_tuning import FineTuningManager

ft_manager = FineTuningManager()
training_file = ft_manager.create_training_data(your_data)
file_id = ft_manager.upload_training_file(training_file)
job_id = ft_manager.create_fine_tuning_job(file_id)
```

## 각 버전별 특징

### 기본 챗봇 (main.py)
- 단순하고 가벼운 인터페이스
- 기본적인 대화 기능
- OpenAI ChatGPT와 동일한 사용감

### PDF 지원 챗봇 (main_with_pdf.py)
- PDF 문서 업로드 및 검색
- 문서 기반 질의응답
- 임베딩을 활용한 유사도 검색

### 고급 UI 챗봇 (main_advanced.py)
- 현대적인 말풍선 스타일 UI
- 비동기 API 호출
- 실시간 상태 표시

## 문제 해결

### PyPDF2 버전 오류
PyPDF2 3.0.0 이상을 사용하는 경우 새로운 API가 적용됩니다. 
현재 코드는 최신 버전과 호환됩니다.

