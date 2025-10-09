"""
표준 RAG Text Splitting 모듈
- RecursiveCharacterTextSplitter를 사용한 의미 단위 청킹
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional


class StandardTextSplitter:
    """표준 RAG 텍스트 분할기"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len,
        separators: Optional[List[str]] = None
    ):
        """
        텍스트 분할기 초기화

        Args:
            chunk_size: 청크 크기 (기본: 1000자)
            chunk_overlap: 청크 간 겹치는 부분 (기본: 200자)
            length_function: 길이 계산 함수
            separators: 분할 기준 문자들
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 기본 분할 기준
        if separators is None:
            separators = [
                "\n\n",  # 문단 구분
                "\n",    # 줄바꿈
                ". ",    # 문장 끝 (영어)
                ".",     # 마침표
                "? ",    # 물음표 뒤 공백
                "?",     # 물음표
                "! ",    # 느낌표 뒤 공백
                "!",     # 느낌표
                " ",     # 공백
                "",      # 마지막 수단
            ]

        # RecursiveCharacterTextSplitter 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=separators,
            keep_separator=True,  # 구분자 유지
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서 리스트를 청크로 분할합니다.

        Args:
            documents: 분할할 Document 객체 리스트

        Returns:
            분할된 Document 객체 리스트
        """
        if not documents:
            print("분할할 문서가 없습니다.")
            return []

        print(f"문서 분할 시작: {len(documents)}개 문서")

        # 문서별 통계
        original_lengths = [len(doc.page_content) for doc in documents]
        total_original_length = sum(original_lengths)

        print(f"   - 원본 총 길이: {total_original_length:,} 문자")
        print(f"   - 평균 문서 길이: {total_original_length // len(documents):,} 문자")

        # 문서 분할 수행
        split_docs = self.text_splitter.split_documents(documents)

        # 분할 결과 통계
        split_lengths = [len(doc.page_content) for doc in split_docs]
        total_split_length = sum(split_lengths)

        print(f"문서 분할 완료:")
        print(f"   - 분할된 청크 수: {len(split_docs)}")
        print(f"   - 총 길이: {total_split_length:,} 문자")
        print(f"   - 평균 청크 크기: {total_split_length // len(split_docs)} 문자")
        print(f"   - 압축률: {len(split_docs) / len(documents):.1f}x")

        # 메타데이터 보강
        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                'chunk_id': i,
                'chunk_size': len(doc.page_content),
                'splitter_type': 'RecursiveCharacterTextSplitter'
            })

        return split_docs

    def split_text(self, text: str) -> List[str]:
        """
        단일 텍스트를 청크로 분할합니다.

        Args:
            text: 분할할 텍스트

        Returns:
            분할된 텍스트 리스트
        """
        if not text.strip():
            print("분할할 텍스트가 없습니다.")
            return []

        print(f"텍스트 분할 시작: {len(text)} 문자")

        # 텍스트 분할
        chunks = self.text_splitter.split_text(text)

        print(f"텍스트 분할 완료: {len(chunks)}개 청크")

        return chunks

    def get_chunk_preview(self, documents: List[Document], num_previews: int = 3) -> None:
        """
        분할된 청크의 미리보기를 출력합니다.

        Args:
            documents: 미리보기할 Document 리스트
            num_previews: 보여줄 청크 수
        """
        if not documents:
            print("미리보기할 문서가 없습니다.")
            return

        print(f"\n청크 미리보기 (상위 {min(num_previews, len(documents))}개):")
        print("=" * 60)

        for i, doc in enumerate(documents[:num_previews]):
            print(f"\n[청크 {i+1}]")
            print(f"크기: {len(doc.page_content)} 문자")
            print(f"소스: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"페이지: {doc.metadata.get('page_number', 'Unknown')}")
            print(f"내용: {doc.page_content[:150]}...")
            if len(doc.page_content) > 150:
                print("(... 더 많은 내용)")

        print("=" * 60)

    def analyze_chunks(self, documents: List[Document]) -> dict:
        """
        청크 분석 정보를 반환합니다.

        Args:
            documents: 분석할 Document 리스트

        Returns:
            청크 분석 결과 딕셔너리
        """
        if not documents:
            return {"message": "분석할 문서가 없습니다."}

        chunk_sizes = [len(doc.page_content) for doc in documents]

        analysis = {
            "total_chunks": len(documents),
            "total_characters": sum(chunk_sizes),
            "average_chunk_size": sum(chunk_sizes) // len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunk_size_distribution": {
                "small (< 500)": len([s for s in chunk_sizes if s < 500]),
                "medium (500-1500)": len([s for s in chunk_sizes if 500 <= s <= 1500]),
                "large (> 1500)": len([s for s in chunk_sizes if s > 1500])
            }
        }

        return analysis


# 사용 예시 및 테스트
if __name__ == "__main__":
    # 테스트 문서 생성
    test_doc = Document(
        page_content="""
        이것은 테스트 문서입니다. 여러 문단으로 구성되어 있습니다.

        첫 번째 문단은 간단한 소개입니다. LangChain과 RAG에 대해 설명합니다.
        RAG는 Retrieval-Augmented Generation의 줄임말입니다.

        두 번째 문단에서는 더 자세한 설명을 제공합니다.
        문서 검색과 생성을 결합한 기술입니다.
        이를 통해 더 정확하고 관련성 높은 답변을 생성할 수 있습니다.

        세 번째 문단은 실제 적용 사례를 다룹니다.
        질의응답 시스템, 문서 요약, 정보 검색 등에 활용됩니다.
        """,
        metadata={"source": "test", "page": 0}
    )

    # 텍스트 분할기 초기화
    splitter = StandardTextSplitter(chunk_size=200, chunk_overlap=50)

    # 문서 분할 테스트
    split_docs = splitter.split_documents([test_doc])

    # 결과 출력
    splitter.get_chunk_preview(split_docs)
    analysis = splitter.analyze_chunks(split_docs)
    print(f"\n분석 결과: {analysis}")