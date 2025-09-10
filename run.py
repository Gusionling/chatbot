#!/usr/bin/env python3
"""
AI 챗봇 실행 스크립트
"""

import sys
import subprocess
import os

def check_requirements():
    """필요한 패키지가 설치되어 있는지 확인"""
    try:
        import PyQt5
        import openai
        return True
    except ImportError as e:
        print(f"필요한 패키지가 설치되어 있지 않습니다: {e}")
        print("다음 명령어로 설치하세요: pip install -r requirements.txt")
        return False

def check_api_key():
    """API 키가 설정되어 있는지 확인"""
    from config import OPENAI_API_KEY
    
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("[WARNING] OpenAI API 키가 설정되지 않았습니다!")
        print("다음 중 하나의 방법으로 API 키를 설정하세요:")
        print("1. .env 파일 생성: cp .env.example .env (그리고 .env 파일 편집)")
        print("2. config.py 파일에서 OPENAI_API_KEY 직접 수정")
        print("3. 환경변수 설정: export OPENAI_API_KEY=your_key_here")
        return False
    
    return True

def main():
    """메인 함수"""
    print(" AI 제품 추천 챗봇을 시작합니다...")
    
    # 요구사항 확인
    if not check_requirements():
        sys.exit(1)
    
    # API 키 확인
    if not check_api_key():
        print("\n계속 진행하시겠습니까? (테스트 모드로 실행됩니다) [y/N]: ", end="")
        if input().lower() != 'y':
            sys.exit(1)
    
    # 메인 애플리케이션 실행
    try:
        from PyQt5.QtWidgets import QApplication
        from main import ChatbotGUI
        
        app = QApplication(sys.argv)
        
        # 앱 스타일 설정
        app.setStyleSheet("""
            QWidget {
                font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
            }
        """)
        
        chatbot = ChatbotGUI()
        chatbot.show()
        
        print("[OK] 챗봇이 성공적으로 시작되었습니다!")
        print("GUI 창을 확인하세요.")
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"[ERROR] 애플리케이션 시작 중 오류가 발생했습니다: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
