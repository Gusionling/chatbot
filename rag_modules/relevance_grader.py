"""
표준 RAG Relevance Grader 모듈
- 문서와 질문/답변의 관련성을 평가
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY, DEFAULT_MODEL
from typing import Literal


# 데이터 모델
class RelevanceScore(BaseModel):
    """관련성 평가 점수"""
    score: Literal["yes", "no"] = Field(
        description="Whether the content is relevant, 'yes' or 'no'"
    )


class RelevanceGrader:
    """
    RAG 문서의 관련성을 평가하는 클래스

    질문-문서, 답변-문서, 질문-답변 간의 관련성을 평가합니다.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0,
        target: Literal["question-retrieval", "answer-retrieval", "question-answer"] = "question-retrieval"
    ):
        """
        관련성 평가기 초기화

        Args:
            model: 사용할 OpenAI 모델
            temperature: 생성 온도 (0-1)
            target: 평가 대상
                - "question-retrieval": 질문과 검색 문서의 관련성
                - "answer-retrieval": 답변과 검색 문서의 관련성
                - "question-answer": 질문과 답변의 관련성
        """
        self.model = model
        self.temperature = temperature
        self.target = target

        # LLM 초기화
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )

        # 구조화된 출력을 위한 LLM
        self.structured_llm = self.llm.with_structured_output(RelevanceScore)

        # 평가 체인 생성
        self.grader_chain = self._create_grader_chain()

        print(f"관련성 평가기 초기화:")
        print(f"   - 모델: {model}")
        print(f"   - 평가 대상: {target}")

    def _create_grader_chain(self):
        """평가 체인을 생성합니다."""

        if self.target == "question-retrieval":
            template = """You are a grader assessing whether a retrieved document is relevant to the given question.

Here is the question:
{question}

Here is the retrieved document:
{context}

If the document contains information that could help answer the question, grade it as relevant.
Consider both semantic meaning and potential usefulness for answering the question.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

Give a binary score 'yes' or 'no' to indicate whether the retrieved document is relevant to the question."""

            input_vars = ["question", "context"]

        elif self.target == "answer-retrieval":
            template = """You are a grader assessing whether a retrieved document is relevant to the given answer.

Here is the answer:
{answer}

Here is the retrieved document:
{context}

If the document contains keyword(s) or semantic meaning related to the answer, grade it as relevant.
The document should support or relate to the information provided in the answer.

Give a binary score 'yes' or 'no' to indicate whether the retrieved document is relevant to the answer."""

            input_vars = ["answer", "context"]

        elif self.target == "question-answer":
            template = """You are a grader assessing whether an answer appropriately addresses the given question.

Here is the question:
{question}

Here is the answer:
{answer}

If the answer directly addresses the question and provides relevant information, grade it as relevant.
Consider both semantic meaning and factual accuracy in your assessment.

Give a binary score 'yes' or 'no' to indicate whether the answer is relevant to the question."""

            input_vars = ["question", "answer"]

        else:
            raise ValueError(f"Invalid target: {self.target}")

        # 프롬프트 생성
        prompt = PromptTemplate(
            template=template,
            input_variables=input_vars,
        )

        # 체인 생성
        chain = prompt | self.structured_llm
        return chain

    def grade(self, **kwargs) -> str:
        """
        관련성을 평가합니다.

        Args:
            **kwargs: 평가에 필요한 입력
                - question-retrieval: question, context
                - answer-retrieval: answer, context
                - question-answer: question, answer

        Returns:
            "yes" 또는 "no"
        """
        try:
            response = self.grader_chain.invoke(kwargs)
            return response.score
        except Exception as e:
            print(f"관련성 평가 실패: {e}")
            # 실패 시 안전하게 "no" 반환
            return "no"

    def is_relevant(self, **kwargs) -> bool:
        """
        관련성을 Boolean으로 반환합니다.

        Args:
            **kwargs: 평가에 필요한 입력

        Returns:
            True (관련 있음) 또는 False (관련 없음)
        """
        score = self.grade(**kwargs)
        return score == "yes"
