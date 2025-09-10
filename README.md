# AI 제품 추천 챗봇

OpenAI GPT 모델을 사용한 제품 추천 챗봇 애플리케이션입니다.
BTS 6기 인텔 최종 프로젝트에 사용했던 코드 입니다. 
<<24.01.25 ~ 24.02.07>>

> local에서 진행을 했던 프로젝트를 보완하여 파이썬 프로젝트로 올립니다.

(기존에 사용하였던 PyPDF2가 버전이 높아지면서 API 가 바뀌어 해당 부분을 수정하고 , 실행파일, 설치파일, 주석을 추가하여 업로드하였습니다.)

## 기능

- PyQt5 기반 GUI 채팅 인터페이스
- 대화 히스토리 관리
- PDF 문서 업로드 및 검색 기능
- OpenAI API 연동
- Fine-tuning 지원

## 설치

필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
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

### 기본 챗봇 실행
```bash
python main.py
```

### PDF 기능 포함 챗봇 실행
```bash
python main_with_pdf.py
```

### 자동 설정 스크립트 사용
```bash
./install.sh
python run.py
```

## 파일 구조

```
chatboot/
├── main.py              # 기본 챗봇
├── main_with_pdf.py     # PDF 기능 포함 챗봇
├── main_advanced.py     # 고급 UI 버전
├── config.py            # 설정 파일
├── pdf_processor.py     # PDF 처리 모듈
├── fine_tuning.py       # Fine-tuning 도구
├── run.py               # 실행 스크립트
├── install.sh           # 설치 스크립트
├── requirements.txt     # 패키지 목록
├── .env.example         # 환경변수 예시
└── README.md           # 프로젝트 설명
```

## Fine-tuning

특정 제품 데이터로 모델을 커스터마이징할 수 있습니다:

```python
from fine_tuning import FineTuningManager

ft_manager = FineTuningManager()
training_file = ft_manager.create_training_data(your_data)
file_id = ft_manager.upload_training_file(training_file)
job_id = ft_manager.create_fine_tuning_job(file_id)
```

## 문제 해결

### PyPDF2 버전 오류
PyPDF2 3.0.0 이상을 사용하는 경우 새로운 API가 적용됩니다. 
현재 코드는 최신 버전과 호환됩니다.
