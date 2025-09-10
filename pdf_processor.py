# PDF 텍스트 추출 및 임베딩 처리
import PyPDF2
import os
from openai import OpenAI
from config import OPENAI_API_KEY
import json
import numpy as np
from typing import List, Dict

class PDFProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.documents = []
        self.embeddings = []
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        PDF 파일에서 텍스트 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            추출된 텍스트
        """
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)  # 최신 API 사용
                text = ""
                
                num_pages = len(reader.pages)  # 최신 API 사용
                print(f"PDF 페이지 수: {num_pages}")
                
                for i in range(num_pages):
                    page = reader.pages[i]  # 최신 API 사용
                    page_text = page.extract_text()  # 최신 API 사용
                    text += page_text + "\n"
                    print(f"페이지 {i+1} 처리 완료")
                
                print("PDF 텍스트 추출 완료!")
                return text
                
        except Exception as e:
            print(f"PDF 읽기 오류: {e}")
            return ""
    
    def split_documents(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        텍스트를 문서 단위로 분할
        
        Args:
            text: 전체 텍스트
            chunk_size: 청크 크기
            
        Returns:
            분할된 문서 리스트
        """
        # 기본적으로 빈 줄로 분할
        documents = text.split("\n\n")
        
        # 너무 긴 문서는 더 작게 분할
        final_documents = []
        for doc in documents:
            if len(doc) > chunk_size:
                # 문장 단위로 분할
                sentences = doc.split(". ")
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) < chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            final_documents.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    final_documents.append(current_chunk.strip())
            else:
                if doc.strip():  # 빈 문서 제외
                    final_documents.append(doc.strip())
        
        print(f"총 {len(final_documents)}개 문서로 분할 완료")
        return final_documents
    
    def create_embeddings(self, documents: List[str]) -> List[List[float]]:
        """
        문서들의 임베딩 생성
        
        Args:
            documents: 문서 리스트
            
        Returns:
            임베딩 리스트
        """
        embeddings = []
        
        print("임베딩 생성 중...")
        for i, doc in enumerate(documents):
            try:
                response = self.client.embeddings.create(
                    input=doc,
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    print(f"{i + 1}/{len(documents)} 임베딩 생성 완료")
                    
            except Exception as e:
                print(f"임베딩 생성 오류 (문서 {i}): {e}")
                # 오류 발생 시 빈 임베딩 추가
                embeddings.append([0.0] * 1536)  # Ada-002 임베딩 크기
        
        print("모든 임베딩 생성 완료!")
        return embeddings
    
    def save_embeddings(self, documents: List[str], embeddings: List[List[float]], 
                       output_file: str = "embeddings.json"):
        """
        문서와 임베딩을 파일로 저장
        
        Args:
            documents: 문서 리스트
            embeddings: 임베딩 리스트
            output_file: 출력 파일명
        """
        data = {
            "documents": documents,
            "embeddings": embeddings
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"임베딩 데이터가 {output_file}에 저장되었습니다.")
    
    def load_embeddings(self, input_file: str = "embeddings.json"):
        """
        저장된 임베딩 데이터 로드
        
        Args:
            input_file: 입력 파일명
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.documents = data["documents"]
            self.embeddings = data["embeddings"]
            
            print(f"{len(self.documents)}개 문서와 임베딩을 로드했습니다.")
            return True
            
        except Exception as e:
            print(f"임베딩 로드 오류: {e}")
            return False
    
    def find_similar_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        쿼리와 유사한 문서 찾기
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            
        Returns:
            유사한 문서들의 리스트
        """
        if not self.documents or not self.embeddings:
            print("임베딩 데이터가 로드되지 않았습니다.")
            return []
        
        try:
            # 쿼리 임베딩 생성
            response = self.client.embeddings.create(
                input=query,
                model="text-embedding-ada-002"
            )
            query_embedding = response.data[0].embedding
            
            # 코사인 유사도 계산
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                similarities.append((similarity, i))
            
            # 유사도 순으로 정렬
            similarities.sort(reverse=True)
            
            # 상위 k개 문서 반환
            results = []
            for similarity, idx in similarities[:top_k]:
                results.append({
                    "document": self.documents[idx],
                    "similarity": similarity,
                    "index": idx
                })
            
            return results
            
        except Exception as e:
            print(f"유사 문서 검색 오류: {e}")
            return []
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        코사인 유사도 계산
        """
        a = np.array(a)
        b = np.array(b)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
    
    def process_pdf(self, pdf_path: str, output_file: str = "embeddings.json"):
        """
        PDF 전체 처리 파이프라인
        
        Args:
            pdf_path: PDF 파일 경로
            output_file: 출력 파일명
        """
        print(f"PDF 처리 시작: {pdf_path}")
        
        # 1. PDF에서 텍스트 추출
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print("텍스트 추출 실패")
            return False
        
        # 2. 문서 분할
        documents = self.split_documents(text)
        if not documents:
            print("문서 분할 실패")
            return False
        
        # 3. 임베딩 생성
        embeddings = self.create_embeddings(documents)
        
        # 4. 저장
        self.save_embeddings(documents, embeddings, output_file)
        
        # 5. 메모리에 로드
        self.documents = documents
        self.embeddings = embeddings
        
        print("PDF 처리 완료!")
        return True

def main():
    """
    예제 실행
    """
    processor = PDFProcessor()
    
    # PDF 파일 경로 (예시)
    pdf_path = "/Users/hyeongkyu/Desktop/app/w5500.pdf"
    
    if os.path.exists(pdf_path):
        # PDF 처리
        success = processor.process_pdf(pdf_path)
        
        if success:
            # 테스트 검색
            query = "W5500이란 무엇인가요?"
            results = processor.find_similar_documents(query)
            
            print(f"\n검색 결과 (쿼리: {query}):")
            for i, result in enumerate(results):
                print(f"\n{i+1}. 유사도: {result['similarity']:.3f}")
                print(f"내용: {result['document'][:200]}...")
    else:
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        print("pdf_path 변수를 실제 PDF 파일 경로로 수정하세요.")

if __name__ == "__main__":
    main()
