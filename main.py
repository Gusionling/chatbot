import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QTextEdit)
from PyQt5.QtCore import Qt
from openai import OpenAI
import os
from config import OPENAI_API_KEY

class ChatbotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.conversation_history = []
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def initUI(self):
        self.setWindowTitle('OpenAI Chatbot')

        # 챗봇 대화 내역을 보여줄 텍스트 에디트
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("background-color: #F0F0F0;")  # 회색 배경 설정

        # 사용자 입력을 받을 라인 에디트
        self.user_input_box = QLineEdit()
        self.user_input_box.returnPressed.connect(self.send_message)

        # 메시지 전송 버튼
        self.send_button = QPushButton('Send')
        self.send_button.clicked.connect(self.send_message)

        # 레이아웃 설정
        vbox = QVBoxLayout()
        vbox.addWidget(self.chat_history)
        vbox.addWidget(self.user_input_box)
        vbox.addWidget(self.send_button)

        self.setLayout(vbox)
        self.resize(400, 500)

    def send_message(self):
        user_input = self.user_input_box.text().strip()
        if user_input == '':
            return
        self.display_message(user_input, "user")
        self.user_input_box.clear()

        self.conversation_history.append({"role": "user", "content": user_input})

        # OpenAI 챗봇에 메시지 전송 및 응답 받기
        try:
            chat_completion = self.client.chat.completions.create(
                messages=self.conversation_history,
                model="gpt-3.5-turbo-1106"
            )
            bot_response = chat_completion.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": bot_response})

            self.display_message(bot_response, "bot")
        except Exception as e:
            self.display_message(f"오류: {str(e)}", "error")

    def display_message(self, message, sender):
        if sender == "user":
            self.chat_history.append(f"사용자: {message}")
        elif sender == "bot":
            self.chat_history.append(f"챗봇: {message}")
        else:
            self.chat_history.append(f"시스템: {message}")
        
        # 스크롤을 맨 아래로 이동
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ChatbotGUI()
    ex.show()
    sys.exit(app.exec_())
