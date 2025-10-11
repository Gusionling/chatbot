"""
표준 RAG Embedding 모듈
- OpenAIEmbeddings를 사용한 캐시 기반 텍스트 임베딩
"""

from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from pathlib import Path
from config import OPENAI_API_KEY


class StandardEmbeddings:
    """표준 RAG 임베딩 처리기"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        cache_dir: str = ".cache/embeddings"
    ):
        """
        임베딩 처리기 초기화

        Args:
            model: OpenAI 임베딩 모델명
            cache_dir: 캐시 디렉토리 경로
        """
        self.model = model
        self.cache_dir = Path(cache_dir)

        # 캐시 기반 임베딩 생성
        self.embeddings = self._create_embedding()

        print(f"임베딩 모델 초기화:")
        print(f"   - 모델: {model}")
        print(f"   - 캐시 디렉토리: {cache_dir}")

    def _create_embedding(self):
        """캐시 기반 OpenAI 임베딩을 생성합니다."""
        try:
            # 캐시 디렉토리 생성
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # 기본 임베딩 모델 생성
            underlying_embeddings = OpenAIEmbeddings(
                model=self.model,
                openai_api_key=OPENAI_API_KEY
            )

            # 파일 기반 캐시 스토어 생성
            store = LocalFileStore(str(self.cache_dir))

            # 캐시 기반 임베딩 생성 (SHA-256 사용으로 보안 강화)
            cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
                underlying_embeddings,
                store,
                namespace=self.model,
                key_encoder="sha256"
            )

            print("캐시 기반 임베딩 초기화 완료")
            return cached_embeddings

        except Exception as e:
            print(f"캐시 임베딩 생성 실패: {e}")
            print("기본 OpenAI 임베딩으로 폴백")
            return OpenAIEmbeddings(
                model=self.model,
                openai_api_key=OPENAI_API_KEY
            )

    def get_embeddings(self):
        """임베딩 객체를 반환합니다."""
        return self.embeddings


# 사용 예시 및 테스트
if __name__ == "__main__":
    try:
        # 임베딩 처리기 초기화
        embedder = StandardEmbeddings()

        # 임베딩 객체 가져오기
        embeddings = embedder.get_embeddings()

        # 테스트 텍스트
        test_texts = [
            "인공지능은 컴퓨터 과학의 한 분야입니다.",
            "머신러닝은 인공지능의 하위 분야입니다.",
        ]

        # 텍스트 임베딩
        print("텍스트 임베딩 테스트:")
        embedded_docs = embeddings.embed_documents(test_texts)
        print(f"임베딩 완료: {len(embedded_docs)}개 문서")

        # 쿼리 임베딩
        query = "AI에 대해 알려주세요"
        query_embedding = embeddings.embed_query(query)
        print(f"쿼리 임베딩 완료: {len(query_embedding)} 차원")

    except Exception as e:
        print(f"테스트 실패: {e}")
        print("API 키를 확인해주세요.")