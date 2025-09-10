import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
# 환경변수 또는 직접 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY_HERE')

# 모델 설정
DEFAULT_MODEL = "gpt-3.5-turbo-1106"
FINE_TUNED_MODEL = None  # Fine-tuned 모델 ID가 있으면 여기에 입력

# GUI 설정
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 700

# 대화 설정
MAX_CONVERSATION_HISTORY = 50  # 최대 대화 히스토리 수
