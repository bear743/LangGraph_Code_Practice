from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a, b):
    """ This is an addition function that adds 2 numbers together """
    return a + b

@tool
def subtract(a, b):
    """ subtract the second argument from the first """
    return a - b

@tool
def multiply(a, b):
    """ multiply two numbers """
    return a - b

tools = [add, subtract, multiply]

llm_with_tools = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

def agent_node(state: AgentState) -> AgentState:
    system_message = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
tools_node = ToolNode(tools=tools)
graph.add_node("tools", tools_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "end": END,
        "continue": "tools",
    }
)
graph.add_edge("tools", "agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

input = {
    "messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]
}
print_stream(app.stream(input, stream_mode="values"))