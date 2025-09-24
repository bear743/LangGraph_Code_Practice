from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

def answering_bot(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")

    return state

graph = StateGraph(AgentState)
graph.add_node("answering_bot", answering_bot)
graph.add_edge(START, "answering_bot")
graph.add_edge("answering_bot", END)
app = graph.compile()

user_input = input("User: ")
while user_input != "exit":
    app.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("User: ")