from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Union

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

def chatbot(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    print(f"AI: {response.content}")

    return state

graph = StateGraph(AgentState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
app =graph.compile()


chat_history = []
user_input = input("User: ")
while user_input != "exit":
    chat_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages": chat_history})
    chat_history = result["messages"]
    user_input = input("User: ")

with open("logging.txt", "w") as file:
    file.write("Conversation Log:\n")

    for message in chat_history:
        if isinstance(message, HumanMessage):
            file.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")
print("Conversation saved to logging.txt")