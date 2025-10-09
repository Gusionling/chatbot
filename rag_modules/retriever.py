"""
표준 RAG Retriever 모듈
- 벡터 저장소 기반 문서 검색
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional
from rag_modules.vector_store import StandardVectorStore


class StandardRetriever:
    """표준 RAG 리트리버"""

    def __init__(
        self,
        vector_store: StandardVectorStore,
        k: int = 4,
        search_type: str = "similarity"
    ):
        """
        리트리버 초기화

        Args:
            vector_store: 벡터 저장소 객체
            k: 반환할 문서 수
            search_type: 검색 유형 ("similarity")
        """
        self.vector_store = vector_store
        self.k = k
        self.search_type = search_type

        # LangChain 리트리버 생성
        self.retriever = self._create_retriever()

        print(f"리트리버 초기화:")
        print(f"   - 검색 유형: {search_type}")
        print(f"   - 반환 문서 수: {k}")

    def _create_retriever(self) -> BaseRetriever:
        """LangChain 리트리버를 생성합니다."""
        if self.vector_store.vectorstore is None:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다.")

        search_kwargs = {"k": self.k}

        return self.vector_store.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        쿼리에 대해 관련 문서를 검색합니다.

        Args:
            query: 검색 쿼리

        Returns:
            검색된 Document 리스트
        """
        if not query.strip():
            print("검색 쿼리가 없습니다.")
            return []

        print(f"문서 검색: '{query[:50]}{'...' if len(query) > 50 else ''}'")

        try:
            # 리트리버를 사용한 문서 검색
            documents = self.retriever.invoke(query)

            print(f"검색 완료: {len(documents)}개 문서")
            for i, doc in enumerate(documents):
                print(f"   {i+1}. 길이: {len(doc.page_content)} 문자")

            return documents

        except Exception as e:
            print(f"문서 검색 실패: {e}")
            return []

    def retrieve_with_score(self, query: str) -> List[tuple]:
        """
        점수와 함께 문서를 검색합니다.

        Args:
            query: 검색 쿼리

        Returns:
            (Document, score) 튜플 리스트
        """
        if not query.strip():
            print("검색 쿼리가 없습니다.")
            return []

        try:
            # 벡터 저장소에서 직접 점수 포함 검색
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=self.k
            )

            return docs_with_scores

        except Exception as e:
            print(f"점수 포함 검색 실패: {e}")
            return []

    def format_documents(self, documents: List[Document]) -> str:
        """
        검색된 문서들을 포맷팅합니다.

        Args:
            documents: Document 리스트

        Returns:
            포맷팅된 문서 텍스트
        """
        if not documents:
            return "검색된 문서가 없습니다."

        formatted_docs = []

        for i, doc in enumerate(documents):
            # 메타데이터에서 정보 추출
            source = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page_number', 'Unknown')

            # 문서 포맷팅
            doc_text = f"[문서 {i+1}]\n"
            doc_text += f"출처: {source}\n"
            doc_text += f"페이지: {page}\n"
            doc_text += f"내용: {doc.page_content}\n"

            formatted_docs.append(doc_text)

        return "\n" + "="*50 + "\n".join(formatted_docs) + "="*50

    def format_context(self, documents: List[Document]) -> str:
        """
        RAG 컨텍스트용으로 문서들을 포맷팅합니다.

        Args:
            documents: Document 리스트

        Returns:
            RAG 컨텍스트 텍스트
        """
        if not documents:
            return "관련 문서를 찾을 수 없습니다."

        context_parts = []

        for i, doc in enumerate(documents):
            # 간단한 컨텍스트 포맷
            context_parts.append(f"문서 {i+1}: {doc.page_content}")

        return "\n\n".join(context_parts)

    def update_k(self, new_k: int) -> None:
        """검색할 문서 수를 업데이트합니다."""
        self.k = new_k
        self.retriever = self._create_retriever()
        print(f"검색 문서 수가 {new_k}개로 업데이트되었습니다.")

    def get_retriever_info(self) -> dict:
        """리트리버 정보를 반환합니다."""
        return {
            "search_type": self.search_type,
            "k": self.k,
            "vector_store_info": self.vector_store.get_store_info()
        }


# 사용 예시 및 테스트
if __name__ == "__main__":
    from rag_modules.embeddings import StandardEmbeddings
    from rag_modules.vector_store import StandardVectorStore

    try:
        # 임베딩과 벡터 저장소 초기화
        embedder = StandardEmbeddings()
        embeddings = embedder.get_embeddings()
        vector_store = StandardVectorStore(embeddings)

        # 테스트 문서들
        test_docs = [
            Document(page_content="인공지능은 컴퓨터가 인간과 같은 지능을 가지도록 하는 기술입니다.", metadata={"source_file": "ai_doc.pdf", "page_number": 1}),
            Document(page_content="머신러닝은 데이터로부터 패턴을 학습하는 인공지능의 한 분야입니다.", metadata={"source_file": "ml_doc.pdf", "page_number": 1}),
            Document(page_content="딥러닝은 인공신경망을 사용하여 복잡한 패턴을 학습하는 기술입니다.", metadata={"source_file": "dl_doc.pdf", "page_number": 1}),
        ]

        # 벡터 저장소 생성
        vector_store.create_vectorstore(test_docs)

        # 리트리버 초기화
        retriever = StandardRetriever(vector_store, k=2)

        # 문서 검색
        query = "인공지능에 대해 알려주세요"
        results = retriever.retrieve(query)

        print(f"\n검색 결과:")
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.page_content}")

        # 포맷팅된 컨텍스트
        context = retriever.format_context(results)
        print(f"\nRAG 컨텍스트:\n{context}")

        # 리트리버 정보
        info = retriever.get_retriever_info()
        print(f"\n리트리버 정보: {info}")

    except Exception as e:
        print(f"테스트 실패: {e}")
        print("API 키와 의존성을 확인해주세요.")