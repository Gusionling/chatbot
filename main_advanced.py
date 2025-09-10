import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QLineEdit, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from openai import OpenAI
import os
from config import OPENAI_API_KEY

class ChatBotWorker(QThread):
    """OpenAI API 호출을 별도 스레드에서 처리"""
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, client, conversation_history):
        super().__init__()
        self.client = client
        self.conversation_history = conversation_history
        
    def run(self):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=self.conversation_history,
                model="gpt-3.5-turbo-1106"  # 또는 fine-tuned 모델 ID 사용
            )
            bot_response = chat_completion.choices[0].message.content
            self.response_ready.emit(bot_response)
        except Exception as e:
            self.error_occurred.emit(str(e))

class ChatbotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.conversation_history = []
        
        # OpenAI 클라이언트 초기화
        try:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"OpenAI 클라이언트 초기화 실패: {str(e)}")
            sys.exit(1)

    def initUI(self):
        self.setWindowTitle('AI Chatbot - 제품 추천 봇')
        self.setGeometry(100, 100, 500, 700)

        # 메인 레이아웃
        layout = QVBoxLayout()

        # 챗봇 대화 내역을 보여줄 텍스트 에디트
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Malgun Gothic', sans-serif;
                font-size: 12px;
            }
        """)

        # 사용자 입력을 받을 라인 에디트
        self.user_input_box = QLineEdit()
        self.user_input_box.setPlaceholderText("메시지를 입력하세요...")
        self.user_input_box.returnPressed.connect(self.send_message)
        self.user_input_box.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 20px;
                font-size: 12px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
        """)

        # 메시지 전송 버튼
        self.send_button = QPushButton('전송')
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)

        # 대화 초기화 버튼
        self.clear_button = QPushButton('대화 초기화')
        self.clear_button.clicked.connect(self.clear_conversation)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 15px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)

        # 버튼 레이아웃
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.send_button)
        button_layout.addWidget(self.clear_button)

        # 레이아웃 설정
        layout.addWidget(self.chat_history)
        layout.addWidget(self.user_input_box)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        
        # 초기 메시지 표시
        self.display_welcome_message()

    def display_welcome_message(self):
        welcome_msg = """
        <div style='text-align: center; padding: 20px; background-color: #e3f2fd; border-radius: 10px; margin: 10px;'>
            <h3 style='color: #1976d2; margin: 0;'> AI 제품 추천 챗봇</h3>
            <p style='color: #666; margin: 10px 0 0 0;'>안녕하세요! 제품에 대해 궁금한 것이 있으시면 언제든 물어보세요.</p>
        </div>
        """
        self.chat_history.insertHtml(welcome_msg)

    def send_message(self):
        user_input = self.user_input_box.text().strip()
        if user_input == '':
            return

        # 사용자 메시지 표시
        self.display_message(user_input, "user")
        self.user_input_box.clear()

        # 대화 히스토리에 추가
        self.conversation_history.append({"role": "user", "content": user_input})

        # 전송 버튼 비활성화
        self.send_button.setEnabled(False)
        self.send_button.setText("답변 생성 중...")

        # 워커 스레드에서 API 호출
        self.worker = ChatBotWorker(self.client, self.conversation_history.copy())
        self.worker.response_ready.connect(self.on_response_ready)
        self.worker.error_occurred.connect(self.on_error_occurred)
        self.worker.start()

    def on_response_ready(self, bot_response):
        # 봇 응답을 대화 히스토리에 추가
        self.conversation_history.append({"role": "assistant", "content": bot_response})
        
        # 봇 메시지 표시
        self.display_message(bot_response, "bot")
        
        # 전송 버튼 다시 활성화
        self.send_button.setEnabled(True)
        self.send_button.setText("전송")

    def on_error_occurred(self, error_message):
        self.display_message(f"오류가 발생했습니다: {error_message}", "error")
        
        # 전송 버튼 다시 활성화
        self.send_button.setEnabled(True)
        self.send_button.setText("전송")

    def display_message(self, message, sender):
        if sender == "user":
            self.chat_history.insertHtml(f'''
                <div style="text-align: right; margin: 10px 0;">
                    <div style="display: inline-block; max-width: 70%; padding: 12px 16px; 
                                background-color: #4CAF50; color: white; border-radius: 18px 18px 5px 18px;
                                font-size: 12px; word-wrap: break-word;">
                        {message}
                    </div>
                </div>
            ''')
        elif sender == "bot":
            self.chat_history.insertHtml(f'''
                <div style="text-align: left; margin: 10px 0;">
                    <div style="display: inline-block; max-width: 70%; padding: 12px 16px; 
                                background-color: white; color: #333; border-radius: 18px 18px 18px 5px;
                                border: 1px solid #ddd; font-size: 12px; word-wrap: break-word;">
                        {message}
                    </div>
                </div>
            ''')
        elif sender == "error":
            self.chat_history.insertHtml(f'''
                <div style="text-align: center; margin: 10px 0;">
                    <div style="display: inline-block; padding: 10px 15px; 
                                background-color: #ffebee; color: #c62828; border-radius: 15px;
                                border: 1px solid #ef5350; font-size: 11px;">
                        ⚠️ {message}
                    </div>
                </div>
            ''')
        
        # 스크롤을 맨 아래로 이동
        self.chat_history.verticalScrollBar().setValue(
            self.chat_history.verticalScrollBar().maximum()
        )

    def clear_conversation(self):
        self.conversation_history = []
        self.chat_history.clear()
        self.display_welcome_message()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 앱 스타일 설정
    app.setStyleSheet("""
        QWidget {
            font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
        }
    """)
    
    try:
        chatbot = ChatbotGUI()
        chatbot.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"애플리케이션 시작 중 오류 발생: {e}")
