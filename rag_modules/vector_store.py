"""
표준 RAG Vector Store 모듈
- FAISS를 사용한 벡터 데이터베이스
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional, Tuple
import os
from pathlib import Path
import hashlib


class StandardVectorStore:
    """표준 RAG 벡터 저장소"""

    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        index_dir: str = ".cache/faiss_index"
    ):
        """
        벡터 저장소 초기화

        Args:
            embeddings: OpenAI 임베딩 객체
            index_dir: FAISS 인덱스 캐시 디렉토리
        """
        self.embeddings = embeddings
        self.index_dir = Path(index_dir)
        self.vectorstore: Optional[FAISS] = None
        self.documents: List[Document] = []

        print(f"벡터 저장소 초기화:")
        print(f"   - 인덱스 디렉토리: {index_dir}")

    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        문서 리스트로부터 FAISS 벡터 저장소를 생성합니다.

        Args:
            documents: Document 객체 리스트

        Returns:
            FAISS 벡터 저장소
        """
        if not documents:
            raise ValueError("벡터 저장소를 생성할 문서가 없습니다.")

        print(f"FAISS 벡터 저장소 생성 시작: {len(documents)}개 문서")

        # 문서 통계
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chars = total_chars // len(documents) if documents else 0

        print(f"   - 총 문자 수: {total_chars:,}")
        print(f"   - 평균 문서 길이: {avg_chars}")

        try:
            # 캐시 확인 시도
            if self._try_load_cached_vectorstore(documents):
                return self.vectorstore

            # 새로운 벡터 저장소 생성
            print("새로운 FAISS 인덱스 생성 중...")
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            self.documents = documents

            # 캐시에 저장
            self._save_vectorstore_cache(documents)

            print(f"FAISS 벡터 저장소 생성 완료")
            print(f"   - 인덱스 크기: {self.vectorstore.index.ntotal}")

            return self.vectorstore

        except Exception as e:
            print(f"벡터 저장소 생성 실패: {e}")
            raise

    def _try_load_cached_vectorstore(self, documents: List[Document]) -> bool:
        """캐시된 벡터 저장소 로드를 시도합니다."""
        try:
            # 인덱스 디렉토리 생성
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # 문서 내용 기반 해시 계산
            doc_contents = "\n".join([doc.page_content for doc in documents])
            doc_hash = hashlib.md5(doc_contents.encode()).hexdigest()

            # 해시 파일과 인덱스 파일 경로
            hash_file = self.index_dir / "doc_hash.txt"
            index_path = str(self.index_dir / "faiss_index")

            # 기존 인덱스가 있고 문서가 변경되지 않았는지 확인
            if (
                hash_file.exists()
                and Path(index_path + ".faiss").exists()
                and hash_file.read_text().strip() == doc_hash
            ):
                # 기존 인덱스 로드
                self.vectorstore = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                self.documents = documents
                print("캐시된 FAISS 인덱스 로드 완료")
                return True

            return False

        except Exception as e:
            print(f"캐시 로드 실패: {e}")
            print("새로운 인덱스를 생성합니다...")
            return False

    def _save_vectorstore_cache(self, documents: List[Document]) -> None:
        """벡터 저장소를 캐시에 저장합니다."""
        try:
            # 문서 해시 계산
            doc_contents = "\n".join([doc.page_content for doc in documents])
            doc_hash = hashlib.md5(doc_contents.encode()).hexdigest()

            # 저장 경로
            hash_file = self.index_dir / "doc_hash.txt"
            index_path = str(self.index_dir / "faiss_index")

            # 인덱스와 해시 저장
            self.vectorstore.save_local(index_path)
            hash_file.write_text(doc_hash)
            print("FAISS 인덱스 캐시 저장 완료")

        except Exception as e:
            print(f"캐시 저장 실패: {e}")
            print("인덱스는 다음 실행 시 캐시되지 않습니다")

    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        유사도 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            유사한 문서 리스트
        """
        if self.vectorstore is None:
            print("벡터 저장소가 초기화되지 않았습니다.")
            return []

        if not query.strip():
            print("검색 쿼리가 없습니다.")
            return []

        print(f"유사도 검색: '{query[:50]}{'...' if len(query) > 50 else ''}'")

        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            print(f"유사도 검색 완료: {len(docs)}개 문서 반환")
            return docs

        except Exception as e:
            print(f"유사도 검색 실패: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        점수와 함께 유사도 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            (문서, 점수) 튜플 리스트
        """
        if self.vectorstore is None:
            print("벡터 저장소가 초기화되지 않았습니다.")
            return []

        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            print(f"점수 포함 검색 완료: {len(docs_with_scores)}개 결과")
            for i, (doc, score) in enumerate(docs_with_scores):
                print(f"   {i+1}. 점수: {score:.4f}")
            return docs_with_scores

        except Exception as e:
            print(f"점수 포함 검색 실패: {e}")
            return []

    def create_retriever(self, search_kwargs: Optional[dict] = None):
        """Retriever 객체를 생성합니다."""
        if self.vectorstore is None:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다.")

        if search_kwargs is None:
            search_kwargs = {"k": 4}

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

        print(f"리트리버 생성 완료: k={search_kwargs.get('k', 4)}")
        return retriever

    def get_store_info(self) -> dict:
        """벡터 저장소 정보를 반환합니다."""
        if self.vectorstore is None:
            return {"message": "벡터 저장소가 초기화되지 않았습니다."}

        return {
            "total_vectors": self.vectorstore.index.ntotal,
            "vector_dimension": self.vectorstore.index.d,
            "total_documents": len(self.documents),
            "store_type": "FAISS",
            "cache_enabled": True,
            "index_dir": str(self.index_dir)
        }


# 사용 예시 및 테스트
if __name__ == "__main__":
    from rag_modules.embeddings import StandardEmbeddings

    try:
        # 임베딩 처리기 초기화
        embedder = StandardEmbeddings()
        embeddings = embedder.get_embeddings()

        # 벡터 저장소 초기화
        vector_store = StandardVectorStore(embeddings)

        # 테스트 문서들
        test_docs = [
            Document(page_content="인공지능은 컴퓨터가 인간과 같은 지능을 가지도록 하는 기술입니다.", metadata={"id": 1}),
            Document(page_content="머신러닝은 데이터로부터 패턴을 학습하는 인공지능의 한 분야입니다.", metadata={"id": 2}),
            Document(page_content="딥러닝은 인공신경망을 사용하여 복잡한 패턴을 학습하는 기술입니다.", metadata={"id": 3}),
        ]

        # 벡터 저장소 생성
        vector_store.create_vectorstore(test_docs)

        # 유사도 검색
        query = "인공지능에 대해 알려주세요"
        results = vector_store.similarity_search(query, k=2)

        print(f"\n검색 결과:")
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.page_content}")

        # 저장소 정보
        info = vector_store.get_store_info()
        print(f"\n저장소 정보: {info}")

    except Exception as e:
        print(f"테스트 실패: {e}")
        print("API 키와 의존성을 확인해주세요.")