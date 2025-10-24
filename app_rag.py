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
from rag_modules.query_rewriter import QueryRewriter

# RAG State 정의
class RAGState(TypedDict):
    """RAG 챗봇의 상태를 정의하는 타입"""
    question: Annotated[str, "사용자 질문"]
    context: Annotated[str, "검색된 문서 컨텍스트"]
    answer: Annotated[str, "생성된 답변"]
    relevance: Annotated[str, "관련성 점수 (yes/no)"]
    messages: Annotated[list, add_messages]
    similarity_scores: Annotated[list, "거리 점수 리스트 (낮을수록 유사)"]

# LLM 설정
llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)

# 표준 RAG 컴포넌트 초기화
document_loader = StandardDocumentLoader()
text_splitter = StandardTextSplitter(chunk_size=1000, chunk_overlap=200)
embedder = StandardEmbeddings()
embeddings = embedder.get_embeddings()
retriever = None  # PDF 로드 후 생성
vectorstore = None  # 벡터 저장소 (거리 점수 계산용)

# 관련성 평가 설정
RELEVANCE_METHOD = "similarity"  # "similarity" 또는 "llm"
SIMILARITY_THRESHOLD = 1.5  # 거리 임계값 (낮을수록 유사, 이 값보다 작으면 관련 있음)

# 관련성 평가기 초기화 (LLM 방식용)
relevance_grader = RelevanceGrader(
    model=DEFAULT_MODEL,
    temperature=0,
    target="question-retrieval"
)

# Query Rewriter 초기화
query_rewriter = QueryRewriter(model=DEFAULT_MODEL, temperature=0)

# RAG 노드 함수들
def query_rewrite_node(state: RAGState) -> RAGState:
    """질문 재작성 노드"""
    original_question = state["question"]

    print("\n" + "="*60)
    print("[QUERY_REWRITE 노드 시작]")
    print("="*60)
    print(f"원본 질문: {original_question}")

    # 질문 재작성
    rewritten_question = query_rewriter.rewrite(original_question)

    print(f"재작성된 질문: {rewritten_question}")
    print("="*60)

    return {"question": rewritten_question}


def retrieve_document(state: RAGState) -> RAGState:
    """문서 검색 노드"""
    global retriever, vectorstore
    question = state["question"]

    print("\n" + "="*60)
    print("[RETRIEVE 노드 시작]")
    print("="*60)
    print(f"질문: {question}")

    similarity_scores = []

    # 리트리버가 초기화되어 있다면 검색 수행
    if retriever is not None and vectorstore is not None:
        try:
            # 유사도 점수와 함께 문서 검색
            docs_with_scores = vectorstore.similarity_search_with_score(question, k=3)

            print(f"\n검색된 문서 개수: {len(docs_with_scores)}")

            documents = []
            # 컨텍스트 포맷팅
            if docs_with_scores:
                # 각 문서 미리보기
                for i, (doc, score) in enumerate(docs_with_scores):
                    documents.append(doc)
                    similarity_scores.append(float(score))

                    print(f"\n--- 문서 {i+1} ---")
                    print(f"거리 점수: {score:.4f} (낮을수록 유사)")
                    print(f"길이: {len(doc.page_content)} 문자")
                    print(f"내용 미리보기: {doc.page_content[:200]}...")
                    if hasattr(doc, 'metadata'):
                        print(f"메타데이터: {doc.metadata}")

                context = "\n\n".join([f"문서 {i+1}: {doc.page_content}" for i, doc in enumerate(documents)])
            else:
                context = "관련 문서를 찾을 수 없습니다."
                print("\n[경고] 검색된 문서가 없습니다.")
        except Exception as e:
            print(f"\n[오류] 문서 검색 오류: {e}")
            context = "문서 검색 중 오류가 발생했습니다."
    else:
        context = "PDF 문서가 로드되지 않았습니다. 일반적인 질문에 답변하겠습니다."
        print("\n[경고] 리트리버가 초기화되지 않음")

    print(f"\n최종 컨텍스트 길이: {len(context)} 문자")
    if similarity_scores:
        print(f"거리 점수 범위: {min(similarity_scores):.4f} ~ {max(similarity_scores):.4f} (낮을수록 유사)")
        print(f"평균 거리 점수: {sum(similarity_scores)/len(similarity_scores):.4f}")
    print("="*60)
    return {"context": context, "similarity_scores": similarity_scores}

def check_relevance(state: RAGState) -> RAGState:
    """관련성 평가 노드"""
    question = state["question"]
    context = state["context"]
    similarity_scores = state.get("similarity_scores", [])

    print("\n" + "="*60)
    print("[CHECK_RELEVANCE 노드 시작]")
    print("="*60)
    print(f"평가 방식: {RELEVANCE_METHOD}")
    print(f"질문: {question}")
    print(f"컨텍스트 길이: {len(context)} 문자")
    print(f"컨텍스트 미리보기: {context[:300]}...")

    # PDF가 로드되지 않았거나 문서가 없으면 관련성을 "no"로 설정
    if "PDF 문서가 로드되지 않았습니다" in context or "관련 문서를 찾을 수 없습니다" in context:
        print("\n[결과] 문서가 없어 평가 스킵 -> LLM 전용 답변")
        print("관련성 점수: no")
        print("="*60)
        return {"relevance": "no"}

    # 관련성 평가 수행
    try:
        if RELEVANCE_METHOD == "similarity":
            # 거리 임계값 기반 평가
            print(f"\n[거리 기반 평가]")
            print(f"거리 임계값: {SIMILARITY_THRESHOLD} (이 값보다 작으면 관련 있음)")

            if not similarity_scores:
                print("[경고] 거리 점수가 없습니다. 'no'로 설정")
                score = "no"
            else:
                # 최소 거리 점수 사용 (FAISS는 거리 기반이므로 낮을수록 유사)
                min_score = min(similarity_scores)
                print(f"거리 점수들: {[f'{s:.4f}' for s in similarity_scores]}")
                print(f"최소 거리 (가장 유사): {min_score:.4f}")

                # FAISS L2 거리는 작을수록 유사함
                # 임계값보다 작으면 관련 있음
                if min_score < SIMILARITY_THRESHOLD:
                    score = "yes"
                    print(f"판단: {min_score:.4f} < {SIMILARITY_THRESHOLD} -> 관련 있음")
                else:
                    score = "no"
                    print(f"판단: {min_score:.4f} >= {SIMILARITY_THRESHOLD} -> 관련 없음")

        elif RELEVANCE_METHOD == "llm":
            # LLM 기반 평가
            print("\n[LLM 기반 평가]")
            print("LLM에게 관련성 평가 요청 중...")
            score = relevance_grader.grade(question=question, context=context)
            print(f"LLM 평가 결과: {score}")

        else:
            print(f"[경고] 알 수 없는 평가 방식: {RELEVANCE_METHOD}")
            score = "no"

        print(f"\n[최종 결과] 관련성 점수: {score}")
        if score == "yes":
            print("다음 경로: RAG 기반 답변 생성")
        else:
            print("다음 경로: LLM 전용 답변 생성")
        print("="*60)
        return {"relevance": score}
    except Exception as e:
        print(f"\n[오류] 관련성 평가 실패: {e}")
        print("관련성 점수: no (기본값)")
        print("="*60)
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

    print("\n" + "="*60)
    print("[RAG_ANSWER 노드 시작]")
    print("="*60)
    print(f"질문: {question}")
    print(f"컨텍스트 길이: {len(context)} 문자")
    print(f"이전 대화 기록: {len(messages)}개 메시지")

    # RAG 프롬프트 구성
    prompt = f"다음 컨텍스트를 참고하여 질문에 답변해주세요.\n\n컨텍스트:\n{context}\n\n질문: {question}\n\n답변:"

    print(f"\n전송할 프롬프트 길이: {len(prompt)} 문자")
    print(f"프롬프트 미리보기:\n{prompt[:500]}...")

    messages_for_llm = messages + [HumanMessage(content=prompt)]

    # LLM 호출
    print("\nLLM에게 답변 생성 요청 중...")
    response = llm.invoke(messages_for_llm)
    answer = response.content

    print(f"\n생성된 답변 길이: {len(answer)} 문자")
    print(f"답변 미리보기: {answer[:200]}...")
    print("="*60)

    return {
        "answer": answer,
        "messages": [("user", question), ("assistant", answer)]
    }


def llm_only_answer(state: RAGState) -> RAGState:
    """LLM 전용 답변 생성 노드 (문서 컨텍스트 미활용)"""
    question = state["question"]
    messages = state["messages"]

    print("\n" + "="*60)
    print("[LLM_ONLY_ANSWER 노드 시작]")
    print("="*60)
    print(f"질문: {question}")
    print(f"이전 대화 기록: {len(messages)}개 메시지")
    print("컨텍스트: 사용 안 함 (순수 LLM 지식 활용)")

    # 일반 프롬프트 구성 (컨텍스트 없이)
    prompt = f"질문: {question}\n\n답변해주세요."

    print(f"\n전송할 프롬프트 길이: {len(prompt)} 문자")
    print(f"프롬프트:\n{prompt}")

    messages_for_llm = messages + [HumanMessage(content=prompt)]

    # LLM 호출
    print("\nLLM에게 답변 생성 요청 중...")
    response = llm.invoke(messages_for_llm)
    answer = response.content

    print(f"\n생성된 답변 길이: {len(answer)} 문자")
    print(f"답변 미리보기: {answer[:200]}...")
    print("="*60)

    return {
        "answer": answer,
        "messages": [("user", question), ("assistant", answer)]
    }

# 그래프 생성
graph_builder = StateGraph(RAGState)

# 노드 추가
graph_builder.add_node("query_rewrite", query_rewrite_node)
graph_builder.add_node("retrieve", retrieve_document)
graph_builder.add_node("check_relevance", check_relevance)
graph_builder.add_node("rag_answer", rag_answer)
graph_builder.add_node("llm_only_answer", llm_only_answer)

# 엣지 정의
graph_builder.add_edge(START, "query_rewrite")
graph_builder.add_edge("query_rewrite", "retrieve")
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
    global retriever, vectorstore

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
    print(f"\n[설정]")
    print(f"관련성 평가 방식: {RELEVANCE_METHOD}")
    if RELEVANCE_METHOD == "similarity":
        print(f"거리 임계값: {SIMILARITY_THRESHOLD} (낮을수록 유사)")
    print(f"LLM 모델: {DEFAULT_MODEL}")

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
