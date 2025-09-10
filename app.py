#!/usr/bin/env python3
"""
간단 실행 스크립트 - 각 버전별 챗봇을 쉽게 실행
"""

import sys
import os

# 경로 설정
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

def run_basic():
    """기본 챗봇 실행"""
    os.chdir(src_path)
    from main import ChatbotGUI
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    chatbot = ChatbotGUI()
    chatbot.show()
    sys.exit(app.exec_())

def run_with_pdf():
    """PDF 기능 포함 챗봇 실행"""
    os.chdir(src_path)
    from main_with_pdf import ChatbotGUI
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    chatbot = ChatbotGUI()
    chatbot.show()
    sys.exit(app.exec_())

def run_advanced():
    """고급 UI 챗봇 실행"""
    os.chdir(src_path)
    from main_advanced import ChatbotGUI
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    chatbot = ChatbotGUI()
    chatbot.show()
    sys.exit(app.exec_())

def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "basic":
            run_basic()
        elif arg == "pdf":
            run_with_pdf()
        elif arg == "advanced":
            run_advanced()
        else:
            print("사용법: python app.py [basic|pdf|advanced]")
    else:
        print("챗봇 실행 옵션을 선택하세요:")
        print("1. 기본 챗봇")
        print("2. PDF 기능 포함")
        print("3. 고급 UI")
        
        choice = input("선택 (1-3): ").strip()
        
        if choice == "1":
            run_basic()
        elif choice == "2":
            run_with_pdf()
        elif choice == "3":
            run_advanced()
        else:
            print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()
