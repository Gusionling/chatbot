"""
표준 RAG Query Rewriter 모듈
- 질문을 벡터 검색에 최적화된 형태로 재작성
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from config import DEFAULT_MODEL


class QueryRewriter:
    """질문 재작성 처리기"""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0):
        """
        Query Rewriter 초기화

        Args:
            model: 사용할 LLM 모델
            temperature: 생성 온도 (0-1)
        """
        self.model = model
        self.temperature = temperature

        # Query Rewrite 프롬프트 정의
        self.prompt = PromptTemplate(
            template="""You are a query reformulation expert. Rewrite the given question to make it more effective for vectorstore retrieval.

# Critical Rules
1. **MAINTAIN THE SAME LANGUAGE**: If the question is in Korean, respond in Korean. If English, respond in English.
2. Output ONLY the rewritten question - no explanations, no translations.

# Steps
1. Identify the core intent and key information needed
2. Enhance clarity and specificity
3. Add relevant keywords for better retrieval
4. Ensure the question is complete and well-formed

# Examples

Input: "인텔 CPU"
Output: "인텔에서 제조한 CPU 프로세서에 대해 설명해주세요"

Input: "인텔 GPU 이름"
Output: "인텔이 개발한 GPU의 이름은 무엇인가요?"

Input: "Intel investment"
Output: "What is the investment amount in Intel?"

Input: "Intel prize 2024"
Output: "What kind of Prizes Intel acheived in 2024?"

# Now rewrite this question:
{question}

Remember: Keep the SAME language as the input question!
""",
            input_variables=["question"],
        )

        # Query Rewriter 체인 생성
        self.chain = self.prompt | ChatOpenAI(model=self.model, temperature=self.temperature) | StrOutputParser()

        print(f"Query Rewriter 초기화:")
        print(f"   - 모델: {model}")
        print(f"   - Temperature: {temperature}")

    def rewrite(self, question: str) -> str:
        """
        질문을 재작성합니다.

        Args:
            question: 원본 질문

        Returns:
            재작성된 질문
        """
        try:
            rewritten_question = self.chain.invoke({"question": question})
            return rewritten_question
        except Exception as e:
            print(f"질문 재작성 실패: {e}")
            # 실패 시 원본 질문 반환
            return question
