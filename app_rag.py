from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

class State(TypedDict):
    """챗봇의 상태를 정의하는 타입

    messages: 대화 메시지 리스트
    - add_messages 함수를 통해 새 메시지가 추가됨 (덮어쓰기가 아닌 추가)
    """
    
    messages:  Annotated[list, add_messages]

#StateGraph 생성
graph_builder = StateGraph(State)

from config import DEFAULT_MODEL

#OpenAI ahepf tkdyd
llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)


# 챗봇 노드 추가
def chatbot(state: State):
    """챗봇 노드 함수

    현재 상태의 메시지를 받아 LLM에 전달하고,
    응답을 새 메시지로 추가하여 반환합니다.
    """
    response = llm.invoke(state["messages"])

    return {"messages":[response]}

graph_builder.add_node("chatbot", chatbot)

# 진입점: 그래프 실행이 시작되는 지점
graph_builder.add_edge(START, "chatbot")

# 종료점: 그래프 실행이 끝나는 지점
graph_builder.add_edge("chatbot", END)

print("실행 흐름: START → chatbot → END")

# 그래프 컴파일
graph = graph_builder.compile()

# 그래프 시각화 함수
def visualize_graph():
    """그래프 구조를 Mermaid 형태로 시각화합니다."""
    try:
        print("\n📊 그래프 구조 (Mermaid):")
        print(graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f"❌ 시각화 오류: {e}")

# 테스트 실행
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    # 그래프 시각화
    visualize_graph()

    # 사용자 입력
    user_input = "안녕하세요! LangGraph에 대해 알려주세요."

    # 그래프 실행
    inputs = {"messages": [HumanMessage(content=user_input)]}

    print(f"\n사용자: {user_input}")
    print("=" * 50)

    # 스트리밍 출력
    for output in graph.stream(inputs):
        for key, value in output.items():
            print(f"노드 '{key}' 실행 결과:")
            if "messages" in value:
                last_message = value["messages"][-1]
                print(f"챗봇: {last_message.content}")

    print("=" * 50)
    print("완료!")
