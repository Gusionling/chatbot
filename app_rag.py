from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from config import DEFAULT_MODEL

# 표준 RAG 모듈 imports
import sys
import os
sys.path.append('rag_modules')
from rag_modules.document_loader import StandardDocumentLoader
from rag_modules.text_splitter import StandardTextSplitter
from rag_modules.embeddings import StandardEmbeddings
from rag_modules import vector_store

# RAG State 정의
class RAGState(TypedDict):
    """RAG 챗봇의 상태를 정의하는 타입"""
    question: Annotated[str, "사용자 질문"]
    context: Annotated[str, "검색된 문서 컨텍스트"]
    answer: Annotated[str, "생성된 답변"]
    messages: Annotated[list, add_messages]

# LLM 설정
llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)

# 표준 RAG 컴포넌트 초기화
document_loader = StandardDocumentLoader()
text_splitter = StandardTextSplitter(chunk_size=1000, chunk_overlap=200)
embedder = StandardEmbeddings()
embeddings = embedder.get_embeddings()
retriever = None  # PDF 로드 후 생성

# RAG 노드 함수들
def retrieve_document(state: RAGState) -> RAGState:
    """문서 검색 노드"""
    global retriever
    question = state["question"]

    # 리트리버가 초기화되어 있다면 검색 수행
    if retriever is not None:
        try:
            # 문서 검색
            documents = retriever.invoke(question)

            # 컨텍스트 포맷팅
            if documents:
                context = "\n\n".join([f"문서 {i+1}: {doc.page_content}" for i, doc in enumerate(documents)])
            else:
                context = "관련 문서를 찾을 수 없습니다."
        except Exception as e:
            print(f"문서 검색 오류: {e}")
            context = "문서 검색 중 오류가 발생했습니다."
    else:
        context = "PDF 문서가 로드되지 않았습니다. 일반적인 질문에 답변하겠습니다."

    return {"context": context}

def llm_answer(state: RAGState) -> RAGState:
    """답변 생성 노드"""
    question = state["question"]
    context = state["context"]

    # 프롬프트 구성
    if "PDF 문서가 로드되지 않았습니다" in context:
        # PDF가 없는 경우 일반 답변
        prompt = f"질문: {question}\n\n답변해주세요."
    else:
        # PDF 기반 답변
        prompt = f"다음 컨텍스트를 참고하여 질문에 답변해주세요.\n\n컨텍스트:\n{context}\n\n질문: {question}\n\n답변:"

    # LLM 호출
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content

    return {
        "answer": answer,
        "messages": [("user", question), ("assistant", answer)]
    }

# 그래프 생성
graph_builder = StateGraph(RAGState)

# 노드 추가
graph_builder.add_node("retrieve", retrieve_document)
graph_builder.add_node("llm_answer", llm_answer)

# 엣지 정의
graph_builder.add_edge("retrieve", "llm_answer")
graph_builder.add_edge("llm_answer", END)

# 진입점 설정
graph_builder.set_entry_point("retrieve")

print("실행 흐름: START → retrieve → llm_answer → END")

# 체크포인터 설정
memory = MemorySaver()

# 그래프 컴파일
graph = graph_builder.compile(checkpointer=memory)

# 그래프 시각화 함수
def visualize_graph():
    """그래프 구조를 Mermaid 형태로 시각화합니다."""
    try:
        print("\n RAG 그래프 구조 (Mermaid):")
        print(graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f" 시각화 오류: {e}")

# PDF 로드 함수
def load_pdf(file_path: str) -> bool:
    """PDF 파일을 로드하고 표준 RAG 파이프라인을 구성합니다."""
    global retriever

    try:
        # 1. 문서 로드
        documents = document_loader.load_pdf(file_path)

        # 2. 텍스트 분할
        split_docs = text_splitter.split_documents(documents)

        # 3. 벡터 저장소 생성
        vectorstore = vector_store.create_vectorstore(split_docs, embeddings)

        # 4. 리트리버 생성
        retriever = vector_store.create_retriever(vectorstore, k=3)

        print(f"RAG 파이프라인 완료: {len(documents)}페이지 → {len(split_docs)}청크")
        return True

    except Exception as e:
        print(f"PDF 로드 실패: {e}")
        return False

# RAG 실행 함수
def run_rag(question: str, config: RunnableConfig = None):
    """RAG 시스템을 실행합니다."""
    if config is None:
        config = RunnableConfig(
            recursion_limit=20,
            configurable={"thread_id": "rag_session_1"}
        )

    # 입력 상태 구성
    inputs = RAGState(question=question)

    print(f" 질문: {question}")
    print("=" * 60)

    # 그래프 실행
    for output in graph.stream(inputs, config):
        for key, value in output.items():
            if key == "retrieve":
                print(f" 문서 검색 완료")
            elif key == "llm_answer":
                print(f" 답변: {value['answer']}")

    print("=" * 60)
    return graph.get_state(config).values

# 테스트
if __name__ == "__main__":
    # 그래프 시각화
    visualize_graph()

    print("\n" + "="*60)
    print(" RAG 시스템 시작")
    print("="*60)

    # 예시 질문들
    questions = [
        "LangGraph에 대해 알려주세요.",
        "PDF 문서가 로드되었나요?",
        "인공지능의 미래는 어떨까요?"
    ]

    # 설정
    config = RunnableConfig(
        recursion_limit=20,
        configurable={"thread_id": "test_session"}
    )

    # 각 질문 실행
    for i, question in enumerate(questions, 1):
        print(f"\n[테스트 {i}]")
        result = run_rag(question, config)

    print("\n 테스트 완료!")
    print(" PDF를 로드하려면 load_pdf('파일경로') 함수를 사용하세요.")
