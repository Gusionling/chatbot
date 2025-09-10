import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QLineEdit, QTextEdit, QFileDialog, QLabel, QHBoxLayout)
from PyQt5.QtCore import Qt
from openai import OpenAI
import os
from config import OPENAI_API_KEY
from pdf_processor import PDFProcessor

class ChatbotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.conversation_history = []
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # PDF 프로세서 초기화
        self.pdf_processor = PDFProcessor()
        self.pdf_loaded = False

    def initUI(self):
        self.setWindowTitle('OpenAI Chatbot with PDF')

        # 메인 레이아웃
        vbox = QVBoxLayout()

        # PDF 로드 섹션
        pdf_layout = QHBoxLayout()
        self.pdf_label = QLabel("PDF: 로드되지 않음")
        self.load_pdf_button = QPushButton('PDF 로드')
        self.load_pdf_button.clicked.connect(self.load_pdf)
        
        pdf_layout.addWidget(self.pdf_label)
        pdf_layout.addWidget(self.load_pdf_button)
        vbox.addLayout(pdf_layout)

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
        vbox.addWidget(self.chat_history)
        vbox.addWidget(self.user_input_box)
        vbox.addWidget(self.send_button)

        self.setLayout(vbox)
        self.resize(500, 600)

    def load_pdf(self):
        """PDF 파일 로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            'PDF 파일 선택', 
            '', 
            'PDF files (*.pdf)'
        )
        
        if file_path:
            self.chat_history.append("시스템: PDF 처리 중... 잠시만 기다려주세요.")
            QApplication.processEvents()  # UI 업데이트
            
            # PDF 처리
            success = self.pdf_processor.process_pdf(file_path)
            
            if success:
                self.pdf_loaded = True
                filename = os.path.basename(file_path)
                self.pdf_label.setText(f"PDF: {filename}")
                self.chat_history.append("시스템: PDF 로드 완료! 이제 PDF 내용에 대해 질문할 수 있습니다.")
            else:
                self.chat_history.append("시스템: PDF 로드 실패.")

    def send_message(self):
        user_input = self.user_input_box.text().strip()
        if user_input == '':
            return
        self.display_message(user_input, "user")
        self.user_input_box.clear()

        # PDF가 로드된 경우 관련 문서 검색
        context = ""
        if self.pdf_loaded:
            similar_docs = self.pdf_processor.find_similar_documents(user_input, top_k=2)
            if similar_docs:
                context = "다음은 관련 문서 내용입니다:\n\n"
                for i, doc in enumerate(similar_docs):
                    context += f"문서 {i+1}: {doc['document']}\n\n"
                context += "위 정보를 바탕으로 답변해주세요.\n\n"

        # 대화 히스토리에 컨텍스트와 함께 추가
        if context:
            full_message = context + f"사용자 질문: {user_input}"
        else:
            full_message = user_input
            
        self.conversation_history.append({"role": "user", "content": full_message})

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
