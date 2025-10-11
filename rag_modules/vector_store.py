"""
표준 RAG Vector Store 모듈
- FAISS를 사용한 벡터 데이터베이스
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
from pathlib import Path
import hashlib


def create_vectorstore(split_docs: List[Document], embeddings, index_dir: str = ".cache/faiss_index"):
    """
    FAISS 벡터 저장소를 생성합니다. (캐싱 지원)

    Args:
        split_docs: 분할된 문서 리스트
        embeddings: 임베딩 객체
        index_dir: 인덱스 캐시 디렉토리

    Returns:
        FAISS 벡터 저장소
    """
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    # 문서 해시 계산
    doc_contents = "\n".join([doc.page_content for doc in split_docs])
    doc_hash = hashlib.md5(doc_contents.encode()).hexdigest()

    hash_file = index_path / "doc_hash.txt"
    faiss_index = str(index_path / "faiss_index")

    # 캐시 확인
    try:
        if (
            hash_file.exists()
            and Path(faiss_index + ".faiss").exists()
            and hash_file.read_text().strip() == doc_hash
        ):
            vectorstore = FAISS.load_local(
                faiss_index,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            print("캐시된 FAISS 인덱스 로드")
            return vectorstore
    except Exception:
        pass

    # 새 인덱스 생성
    vectorstore = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )

    # 캐시 저장
    try:
        vectorstore.save_local(faiss_index)
        hash_file.write_text(doc_hash)
    except Exception:
        pass

    return vectorstore


def create_retriever(vectorstore, k: int = 4):
    """
    Retriever를 생성합니다.

    Args:
        vectorstore: FAISS 벡터 저장소
        k: 검색할 문서 수

    Returns:
        Retriever 객체
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
