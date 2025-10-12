from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from config import DEFAULT_MODEL

# 표준 RAG 모듈 imports
import sys
import os

# OpenMP 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('rag_modules')
from rag_modules.document_loader import StandardDocumentLoader
from rag_modules.text_splitter import StandardTextSplitter
from rag_modules.embeddings import StandardEmbeddings
from rag_modules import vector_store
from rag_modules.relevance_grader import RelevanceGrader

# RAG State 정의
class RAGState(TypedDict):
    """RAG 챗봇의 상태를 정의하는 타입"""
    question: Annotated[str, "사용자 질문"]
    context: Annotated[str, "검색된 문서 컨텍스트"]
    answer: Annotated[str, "생성된 답변"]
    relevance: Annotated[str, "관련성 점수 (yes/no)"]
    messages: Annotated[list, add_messages]

# LLM 설정
llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)

# 표준 RAG 컴포넌트 초기화
document_loader = StandardDocumentLoader()
text_splitter = StandardTextSplitter(chunk_size=1000, chunk_overlap=200)
embedder = StandardEmbeddings()
embeddings = embedder.get_embeddings()
retriever = None  # PDF 로드 후 생성

# 관련성 평가기 초기화
relevance_grader = RelevanceGrader(
    model=DEFAULT_MODEL,
    temperature=0,
    target="question-retrieval"
)

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

def check_relevance(state: RAGState) -> RAGState:
    """관련성 평가 노드"""
    question = state["question"]
    context = state["context"]

    # PDF가 로드되지 않았거나 문서가 없으면 관련성을 "no"로 설정
    if "PDF 문서가 로드되지 않았습니다" in context or "관련 문서를 찾을 수 없습니다" in context:
        print("[관련성 평가] 문서가 없어 평가 스킵 -> LLM 전용 답변")
        return {"relevance": "no"}

    # 관련성 평가 수행
    try:
        score = relevance_grader.grade(question=question, context=context)
        print(f"[관련성 평가] 결과: {score}")
        return {"relevance": score}
    except Exception as e:
        print(f"[관련성 평가] 오류: {e}")
        return {"relevance": "no"}


def decide_path(state: RAGState) -> str:
    """조건부 분기: 관련성에 따라 경로 결정"""
    relevance = state.get("relevance", "no")

    if relevance == "yes":
        print("[경로 결정] RAG 기반 답변 생성")
        return "rag_answer"
    else:
        print("[경로 결정] LLM 전용 답변 생성")
        return "llm_only_answer"


def rag_answer(state: RAGState) -> RAGState:
    """RAG 기반 답변 생성 노드 (문서 컨텍스트 활용)"""
    question = state["question"]
    context = state["context"]
    messages = state["messages"]

    # RAG 프롬프트 구성
    prompt = f"다음 컨텍스트를 참고하여 질문에 답변해주세요.\n\n컨텍스트:\n{context}\n\n질문: {question}\n\n답변:"

    messages_for_llm = messages + [HumanMessage(content=prompt)]

    # LLM 호출
    response = llm.invoke(messages_for_llm)
    answer = response.content

    return {
        "answer": answer,
        "messages": [("user", question), ("assistant", answer)]
    }


def llm_only_answer(state: RAGState) -> RAGState:
    """LLM 전용 답변 생성 노드 (문서 컨텍스트 미활용)"""
    question = state["question"]
    messages = state["messages"]

    # 일반 프롬프트 구성 (컨텍스트 없이)
    prompt = f"질문: {question}\n\n답변해주세요."

    messages_for_llm = messages + [HumanMessage(content=prompt)]

    # LLM 호출
    response = llm.invoke(messages_for_llm)
    answer = response.content

    return {
        "answer": answer,
        "messages": [("user", question), ("assistant", answer)]
    }

# 그래프 생성
graph_builder = StateGraph(RAGState)

# 노드 추가
graph_builder.add_node("retrieve", retrieve_document)
graph_builder.add_node("check_relevance", check_relevance)
graph_builder.add_node("rag_answer", rag_answer)
graph_builder.add_node("llm_only_answer", llm_only_answer)

# 엣지 정의
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "check_relevance")

# 조건부 엣지: 관련성에 따라 경로 분기
graph_builder.add_conditional_edges(
    "check_relevance",
    decide_path,
    {
        "rag_answer": "rag_answer",
        "llm_only_answer": "llm_only_answer"
    }
)

# 양쪽 경로 모두 END로 연결
graph_builder.add_edge("rag_answer", END)
graph_builder.add_edge("llm_only_answer", END)

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

# 문서 로드 함수
def load_document(file_path: str) -> bool:
    """문서 파일을 로드하고 표준 RAG 파이프라인을 구성합니다."""
    global retriever

    try:
        # 파일 확장자에 따라 로드 방식 결정
        if file_path.endswith('.pdf'):
            documents = document_loader.load_pdf(file_path)
        elif file_path.endswith('.txt') or file_path.endswith('.md'):
            # 텍스트 파일 로드
            documents = document_loader.load_text_file(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path}")

        # 2. 텍스트 분할
        split_docs = text_splitter.split_documents(documents)

        # 3. 벡터 저장소 생성
        vectorstore = vector_store.create_vectorstore(split_docs, embeddings)

        # 4. 리트리버 생성
        retriever = vector_store.create_retriever(vectorstore, k=3)

        print(f"RAG 파이프라인 완료: {len(documents)}개 문서 → {len(split_docs)}청크")
        return True

    except Exception as e:
        print(f"문서 로드 실패: {e}")
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
                print(f"[문서 검색] 완료")
            elif key == "rag_answer":
                print(f"[답변] {value['answer']}")
            elif key == "llm_only_answer":
                print(f"[답변] {value['answer']}")

    print("=" * 60)
    return graph.get_state(config).values

# 챗봇 실행
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" 인텔 제품 RAG 챗봇")
    print("="*60)

    # 문서 로드
    doc_path = "data/intel_namu.txt"
    print(f"\n문서 로딩 중: {doc_path}")

    if load_document(doc_path):
        print("문서 로드 성공! 인텔 제품에 대해 질문해주세요.\n")
    else:
        print("문서 로드 실패. 일반 모드로 실행합니다.\n")

    # 설정
    config = RunnableConfig(
        recursion_limit=20,
        configurable={"thread_id": "chatbot_session"}
    )

    print("대화를 시작합니다. (종료: 'exit', 'quit', '종료')")
    print("="*60 + "\n")

    # 대화형 루프
    while True:
        try:
            question = input("질문: ").strip()

            if not question:
                continue

            if question.lower() in ['exit', 'quit', '종료', 'q']:
                print("\n챗봇을 종료합니다.")
                break

            print()
            run_rag(question, config)
            print()

        except KeyboardInterrupt:
            print("\n\n챗봇을 종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}\n")
