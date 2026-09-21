
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, MessagesState
from IPython.display import display, Image
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY_NEW")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY_NEW is missing in the .env file")

grok_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY
)


def add(a:int, b:int) -> int:
    """
        Add two numbers a and b
        a: int
        b: int
    """
    return a + b

def divide(a:int, b:int) -> int:
    """
        Add two numbers a and b
        a: int
        b: int
    """
    return a / b

def subtract(a:int, b:int) -> int:
    """
        Subtract two numbers a and b
        a: int
        b: int
    """
    return a - b

tools = [add, subtract, divide]

llm_with_tools = grok_llm.bind_tools(tools)


def agent(state: MessagesState) -> MessagesState:
    return ({"messages": llm_with_tools.invoke(state.get("messages"))})


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

# query = "Add 10 with 20, then divide it by 2 and subtract it by 5"
# res = graph.invoke({"messages": HumanMessage(content="Add 10 with 20")})
# res = graph.invoke({"messages": HumanMessage(content="Subtract it by 8")})
# res = graph.invoke({"messages": HumanMessage(content=query)})


# memory = MemorySaver()
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

graph = builder.compile(checkpointer=memory)

# config={"configurable":{"thread_id": "raj1"}}
# # config_1={"configurable":{"thread_id": "ab1"}}

# res = graph.invoke({"messages": HumanMessage(content="Add it with 3")}, config=config)
# res = graph.invoke({"messages": HumanMessage(content="Subtract it by 2")}, config=config)


# for msg in res.get('messages'):
#     msg.pretty_print()

# display(Image(graph.get_graph().draw_mermaid_png()))

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "raj1"}}
    res = graph.invoke({"messages": HumanMessage(content="Add it with 3")}, config=config)
    res = graph.invoke({"messages": HumanMessage(content="Subtract it by 2")}, config=config)
    for msg in res.get('messages'):
        msg.pretty_print()
    display(Image(graph.get_graph().draw_mermaid_png()))