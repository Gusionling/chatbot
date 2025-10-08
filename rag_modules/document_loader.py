"""
표준 RAG Document Loading 모듈
- PDFPlumberLoader를 사용한 고품질 PDF 텍스트 추출
"""

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from typing import List, Optional
from pathlib import Path
import os


class StandardDocumentLoader:
    """표준 RAG 문서 로더"""

    def __init__(self):
        self.loaded_documents: List[Document] = []
        self.source_files: List[str] = []

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        PDFPlumberLoader를 사용하여 PDF 문서를 로드합니다.

        Args:
            file_path: PDF 파일 경로

        Returns:
            Document 객체 리스트
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

            # PDF 파일 확인
            if not file_path.lower().endswith('.pdf'):
                raise ValueError(f"PDF 파일이 아닙니다: {file_path}")

            print(f"📄 PDF 로딩 시작: {file_path}")

            # PDFPlumberLoader 사용 (고품질 텍스트 추출)
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()

            if not documents:
                raise ValueError(f"문서 내용을 추출할 수 없습니다: {file_path}")

            # 메타데이터 추가
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source_file': file_path,
                    'file_name': Path(file_path).name,
                    'page_number': i,
                    'loader_type': 'PDFPlumberLoader'
                })

            print(f"✅ PDF 로딩 완료: {len(documents)}개 페이지")
            return documents

        except Exception as e:
            print(f"❌ PDF 로딩 실패: {e}")
            raise

    def load_multiple_pdfs(self, file_paths: List[str]) -> List[Document]:
        """
        여러 PDF 파일을 로드합니다.

        Args:
            file_paths: PDF 파일 경로 리스트

        Returns:
            모든 문서의 Document 객체 리스트
        """
        all_documents = []
        successful_files = []
        failed_files = []

        for file_path in file_paths:
            try:
                documents = self.load_pdf(file_path)
                all_documents.extend(documents)
                successful_files.append(file_path)
            except Exception as e:
                print(f"⚠️ 파일 로딩 실패: {file_path} - {e}")
                failed_files.append(file_path)
                continue

        # 결과 요약
        print(f"\n📊 로딩 결과:")
        print(f"✅ 성공: {len(successful_files)}개 파일")
        print(f"❌ 실패: {len(failed_files)}개 파일")
        print(f"📄 총 문서: {len(all_documents)}개 페이지")

        if failed_files:
            print(f"실패한 파일들: {failed_files}")

        self.loaded_documents = all_documents
        self.source_files = successful_files

        return all_documents

    def get_document_info(self) -> dict:
        """로드된 문서 정보를 반환합니다."""
        if not self.loaded_documents:
            return {"message": "로드된 문서가 없습니다."}

        return {
            "total_documents": len(self.loaded_documents),
            "source_files": self.source_files,
            "total_characters": sum(len(doc.page_content) for doc in self.loaded_documents),
            "document_types": list(set(doc.metadata.get('loader_type', 'unknown')
                                     for doc in self.loaded_documents))
        }


# 사용 예시 및 테스트
if __name__ == "__main__":
    loader = StandardDocumentLoader()

    # 테스트용 (실제 PDF 파일 경로로 변경 필요)
    test_file = "data/sample.pdf"

    if os.path.exists(test_file):
        try:
            documents = loader.load_pdf(test_file)
            print(f"로드된 문서 수: {len(documents)}")
            print(f"첫 번째 페이지 미리보기: {documents[0].page_content[:200]}...")
        except Exception as e:
            print(f"테스트 실패: {e}")
    else:
        print("테스트 파일이 없습니다. 실제 PDF 파일로 테스트해주세요.")