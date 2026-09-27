import os

from dotenv import load_dotenv

import sqlite3
from functools import partial
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langgraph.graph import StateGraph, START, MessagesState,END
from langchain_core.vectorstores import InMemoryVectorStore
from groq import RateLimitError, APIStatusError, APIConnectionError
from typing import Annotated, Optional, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, ToolMessage



load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY_NEW")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

groq_llm = ChatGroq(
    temperature=0,
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    reasoning_format="hidden"
)


pdf_loader = PyPDFLoader("C:\\Users\\C901925\\Desktop\\lang graph project\\AcmeNova_Company_Knowledge_Base.pdf")

pdf_docs = pdf_loader.load()

print("PDF pages:", len(pdf_docs))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(pdf_docs)

print("Total chunks:", len(split_docs))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = InMemoryVectorStore.from_documents(documents=split_docs, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print("\nConnecting to database...")

db = SQLDatabase.from_uri(DATABASE_URL)

print("Database connected.")



@tool
def search_documents(input: str) -> str:
    """
    Search the document knowledge base.

    Use this tool for questions related to:

    - AcmeNova company information (head office, employees, policies, etc.)
    - Coolie movie

    The answer should be based on the retrieved documents.
    """

    print("\n[RAG TOOL]")
    print("Question:", input)

    documents = retriever.invoke(input)

    if not documents:
        return "No relevant documents were found."

    context = "\n\n".join(document.page_content for document in documents)

    print("Documents retrieved:", len(documents))

    return context


tavily_search = TavilySearch(api_key=TAVILY_API_KEY, max_results=3)


@tool
def web_search(input: str) -> str:
    """
    Search the internet for current information.

    Use this tool when the question requires:

    - current information
    - recent news
    - information not available in the document database
    """

    print("\n[WEB SEARCH TOOL]")
    print("Question:", input)

    return tavily_search.invoke({"query": input})


# Direct SQL chain avoids create_sql_agent's tool_choice="none" conflict with Groq reasoning models.
def _generate_sql(question: str) -> str:
    schema = db.get_table_info()
    prompt = (
        "You are a SQLite expert. Given the schema below, write ONE valid SQLite SELECT query "
        "that answers the question. Return ONLY the raw SQL with no markdown, no backticks, "
        "no explanation.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\nSQL:"
    )
    sql = groq_llm.invoke(prompt).content.strip()
    return sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()


@tool
def search_database(input: str) -> str:
    """
    Search the users database.

    Use this tool for questions about:

    - user information
    - username
    - name
    - email
    - date of birth
    - account status
    - account creation date
    - number of users
    """

    print("\n[SQL TOOL]")
    print("Question:", input)

    sql = _generate_sql(input)
    print("Generated SQL:", sql)

    if not sql.lower().lstrip().startswith("select"):
        return "Only read-only SELECT queries are permitted."

    try:
        result = db.run(sql)
    except Exception as exc:
        return f"SQL execution failed: {exc}"

    print("Result:", result)

    return f"Query: {sql}\nResult: {result}"


tools = [search_documents, search_database, web_search]


# class State(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

class AgentState(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    intermediate_steps: list[str]
    tool_output: Optional[str]
    final_answer: Optional[str]
    next_step: str


def agent_node(state: AgentState):

    print("\n==============================")
    print("LLM NODE")
    print("==============================")

    messages = state["messages"]
    llm_with_tools = groq_llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)

    intermediate_steps = list(state.get("intermediate_steps") or [])

    if response.tool_calls:
        print(f"\nLLM Agent requested tools: \n{(response.tool_calls)}")
        print(f"[DEBUG] Total tool_calls in this turn: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"\n\nTool: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")
            intermediate_steps.append(f"Called {tool_call['name']} with {tool_call['args']}")
        next_step = response.tool_calls[0]["name"]
        print(f"\n\n[DEBUG] next_step set to FIRST tool only -> '{next_step}'")
        # if len(response.tool_calls) > 1:
        #     skipped = [tc["name"] for tc in response.tool_calls[1:]]
        #     print(f"[DEBUG] These tool_calls will NOT run this cycle: {skipped}")
        final_answer = None
    else:
        print("\nLLM generated final answer.")
        next_step = "end"
        final_answer = response.content

    print(f"[DEBUG] agent_node returning next_step='{next_step}'")

    return {
        "messages": [response],
        "intermediate_steps": intermediate_steps,
        "next_step": next_step,
        "final_answer": final_answer,
    }


tools_by_name = {t.name: t for t in tools}


def _run_tool_node(state: AgentState, tool_name: str):

    print(f"\n[DEBUG] Entered tool node for: '{tool_name}'")

    last_message = state["messages"][-1]
    matching_calls = [tc for tc in last_message.tool_calls if tc["name"] == tool_name]
   
    print(f"[DEBUG] Matching calls to execute now: {[tc['name'] for tc in matching_calls]}")
    
    tool_messages = []
    for tool_call in matching_calls:
        output = tools_by_name[tool_name].invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))

    tool_output = tool_messages[-1].content if tool_messages else None
    intermediate_steps = list(state.get("intermediate_steps") or [])
    intermediate_steps.append(f"{tool_name} output: {tool_output}")

    print(f"[DEBUG] tool node '{tool_name}' done, routing back to agent_node")

    return {
        "messages": tool_messages,
        "tool_output": tool_output,
        "intermediate_steps": intermediate_steps,
        "next_step": "agent_node",
    }


def search_documents_node(state: AgentState):
    return _run_tool_node(state, "search_documents")


def search_database_node(state: AgentState):
    return _run_tool_node(state, "search_database")


def web_search_node(state: AgentState):
    return _run_tool_node(state, "web_search")


def route_next(state: AgentState):

    choice = state["next_step"]
    print(f"\n[DEBUG] route_next() sees next_step='{choice}'")

    if choice == "search_documents":
        print("[DEBUG] Decision key 'route_docs' -> maps to node 'search_documents_node'")
        return "route_docs"

    elif choice == "search_database":
        print("[DEBUG] Decision key 'route_db' -> maps to node 'search_database_node'")
        return "route_db"

    elif choice == "web_search":
        print("[DEBUG] Decision key 'route_web' -> maps to node 'web_search_node'")
        return "route_web"

    print("[DEBUG] Decision key END -> maps to node END")
    return END


builder = StateGraph(AgentState)

builder.add_node("agent_node", agent_node)
builder.add_node("search_documents_node", search_documents_node)
builder.add_node("search_database_node", search_database_node)
builder.add_node("web_search_node", web_search_node)

builder.add_edge(START, "agent_node")
builder.add_conditional_edges(
    "agent_node",
    route_next,
    {
        # decision key returned by route_next -> actual node name registered above
        "route_docs": "search_documents_node",
        "route_db": "search_database_node",
        "route_web": "web_search_node",
        END: END,
    },
)
builder.add_edge("search_documents_node", "agent_node")
builder.add_edge("search_database_node", "agent_node")
builder.add_edge("web_search_node", "agent_node")


# graph = builder.compile()
conn = sqlite3.connect("rag_checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory)

print("\n===================================")
print("LANGGRAPH CREATED SUCCESSFULLY")
print("===================================")


def ask_question(question: str, thread_id: str = "default"):

    print("\n")
    print("==============================================")
    print("USER QUESTION")
    print("==============================================")
    print(question)

    config = {"configurable": {"thread_id": thread_id}}

    existing_state = graph.get_state(config)
    messages = []
    if not existing_state.values.get("messages"):
        messages.append({
            "role": "system",
            "content": (
                "Answer using the tool results you already have. "
                "Do not call the same tool more than once with a similar query, and do not call "
                "both tools for the same sub-question if one tool already returned a clear answer. "
                "Use search_documents ONLY for AcmeNova company facts (head office, employees, policies). "
                "Use search_database for questions about users, accounts, emails, or signup data. "
                "Use web_search for any real-world person's name (e.g. Vijay) that is not an AcmeNova "
                "employee listed in search_documents results — do not call search_documents again for that name."
            )
        })
    messages.append({"role": "user", "content": question})

    try:
        result = graph.invoke({"query": question, "messages": messages}, config=config)
        final_message = result["messages"][-1]
        answer = final_message.content

    except RateLimitError as exc:
        answer = (
            "The AI service has hit its rate/token limit for now. "
            f"Please try again shortly.\n\nDetails: {exc}"
        )
    except APIConnectionError as exc:
        answer = f"Could not reach the AI service (network issue). Details: {exc}"
    except APIStatusError as exc:
        answer = f"The AI service returned an error (status {exc.status_code}). Details: {exc}"
    except Exception as exc:
        answer = f"Unexpected error while generating the answer: {exc}"

    print("\n")
    print("==============================================")
    print("FINAL ANSWER")
    print("==============================================")
    print(answer)

    return answer


if __name__ == "__main__":
    # question = "latest ipl winner 2026 and the AcmeNova head office location and \"How many inactive users?\""
    # question = "How many users are there?"
    # question = "latest ipl winner 2026 and the AcmeNova head office location"
    question = "Who manages Cloud Engineering?"
    thread_id="demo2"
    ask_question(question, thread_id=thread_id)







# import os

# from dotenv import load_dotenv

# import sqlite3
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_community.vectorstores import FAISS

# from langchain_core.tools import tool
# from langchain_tavily import TavilySearch
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import create_sql_agent
# from langgraph.graph import StateGraph, START, MessagesState,END
# from langchain_core.vectorstores import InMemoryVectorStore
# from groq import RateLimitError, APIStatusError, APIConnectionError
# from typing import Annotated, Optional, TypedDict
# from langgraph.graph.message import add_messages
# from langchain_core.messages import AnyMessage, ToolMessage



# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY_NEW")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# DATABASE_URL = os.getenv("DATABASE_URL")

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# groq_llm = ChatGroq(
#     temperature=0,
#     model="openai/gpt-oss-120b",
#     api_key=GROQ_API_KEY,
#     reasoning_format="hidden"
# )


# pdf_loader = PyPDFLoader("C:\\Users\\C901925\\Desktop\\lang graph project\\AcmeNova_Company_Knowledge_Base.pdf")

# pdf_docs = pdf_loader.load()

# print("PDF pages:", len(pdf_docs))

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# split_docs = text_splitter.split_documents(pdf_docs)

# print("Total chunks:", len(split_docs))

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vectorstore = InMemoryVectorStore.from_documents(documents=split_docs, embedding=embeddings)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# print("\nConnecting to database...")

# db = SQLDatabase.from_uri(DATABASE_URL)

# print("Database connected.")



# @tool
# def search_documents(input: str) -> str:
#     """
#     Search the document knowledge base.

#     Use this tool for questions related to:

#     - AcmeNova company information (head office, employees, policies, etc.)
#     - Coolie movie

#     The answer should be based on the retrieved documents.
#     """

#     print("\n[RAG TOOL]")
#     print("Question:", input)

#     documents = retriever.invoke(input)

#     if not documents:
#         return "No relevant documents were found."

#     context = "\n\n".join(document.page_content for document in documents)

#     print("Documents retrieved:", len(documents))

#     return context


# tavily_search = TavilySearch(api_key=TAVILY_API_KEY, max_results=3)


# @tool
# def web_search(input: str) -> str:
#     """
#     Search the internet for current information.

#     Use this tool when the question requires:

#     - current information
#     - recent news
#     - information not available in the document database
#     """

#     print("\n[WEB SEARCH TOOL]")
#     print("Question:", input)

#     return tavily_search.invoke({"query": input})


# # Direct SQL chain avoids create_sql_agent's tool_choice="none" conflict with Groq reasoning models.
# def _generate_sql(question: str) -> str:
#     schema = db.get_table_info()
#     prompt = (
#         "You are a SQLite expert. Given the schema below, write ONE valid SQLite SELECT query "
#         "that answers the question. Return ONLY the raw SQL with no markdown, no backticks, "
#         "no explanation.\n\n"
#         f"Schema:\n{schema}\n\n"
#         f"Question: {question}\n\nSQL:"
#     )
#     sql = groq_llm.invoke(prompt).content.strip()
#     return sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()


# @tool
# def search_database(input: str) -> str:
#     """
#     Search the users database.

#     Use this tool for questions about:

#     - user information
#     - username
#     - name
#     - email
#     - date of birth
#     - account status
#     - account creation date
#     - number of users
#     """

#     print("\n[SQL TOOL]")
#     print("Question:", input)

#     sql = _generate_sql(input)
#     print("Generated SQL:", sql)

#     if not sql.lower().lstrip().startswith("select"):
#         return "Only read-only SELECT queries are permitted."

#     try:
#         result = db.run(sql)
#     except Exception as exc:
#         return f"SQL execution failed: {exc}"

#     print("Result:", result)

#     return f"Query: {sql}\nResult: {result}"


# tools = [search_documents, search_database, web_search]


# # class State(TypedDict):
# #     messages: Annotated[list[AnyMessage], add_messages]

# class AgentState(TypedDict):
#     query: str
#     messages: Annotated[list, add_messages]
#     intermediate_steps: list[str]
#     tool_output: Optional[str]
#     final_answer: Optional[str]
#     next_step: str


# def agent_node(state: AgentState):

#     print("\n==============================")
#     print("LLM NODE")
#     print("==============================")

#     messages = state["messages"]
#     llm_with_tools = groq_llm.bind_tools(tools)
#     response = llm_with_tools.invoke(messages)

#     print("response.tool_calls:", response.tool_calls)

#     intermediate_steps = list(state.get("intermediate_steps") or [])

#     if response.tool_calls:
#         print("\nLLM requested tools:")
#         for tool_call in response.tool_calls:
#             print("Tool:", tool_call["name"])
#             print("Arguments:", tool_call["args"])
#             intermediate_steps.append(f"Called {tool_call['name']} with {tool_call['args']}")
#         next_step = "tools_node"
#         final_answer = None
#     else:
#         print("\nLLM generated final answer.")
#         next_step = "end"
#         final_answer = response.content

#     return {
#         "messages": [response],
#         "intermediate_steps": intermediate_steps,
#         "next_step": next_step,
#         "final_answer": final_answer,
#     }


# tools_by_name = {t.name: t for t in tools}


# def tools_node(state: AgentState):
#     """Runs every tool call from the last AIMessage in one pass, regardless of tool name."""

#     last_message = state["messages"][-1]

#     tool_messages = []
#     intermediate_steps = list(state.get("intermediate_steps") or [])
#     for tool_call in last_message.tool_calls:
#         output = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
#         tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
#         intermediate_steps.append(f"{tool_call['name']} output: {output}")

#     tool_output = tool_messages[-1].content if tool_messages else None

#     return {
#         "messages": tool_messages,
#         "tool_output": tool_output,
#         "intermediate_steps": intermediate_steps,
#         "next_step": "agent_node",
#     }


# def route_next(state: AgentState):
#     return "tools_node" if state["next_step"] == "tools_node" else END


# builder = StateGraph(AgentState)

# builder.add_node("agent_node", agent_node)
# builder.add_node("tools_node", tools_node)

# builder.add_edge(START, "agent_node")
# builder.add_conditional_edges(
#     "agent_node",
#     route_next,
#     {
#         "tools_node": "tools_node",
#         END: END,
#     },
# )
# builder.add_edge("tools_node", "agent_node")


# # graph = builder.compile()
# conn = sqlite3.connect("rag_checkpoints.sqlite", check_same_thread=False)
# memory = SqliteSaver(conn)
# graph = builder.compile(checkpointer=memory)

# print("\n===================================")
# print("LANGGRAPH CREATED SUCCESSFULLY")
# print("===================================")


# def ask_question(question: str, thread_id: str = "default"):

#     print("\n")
#     print("==============================================")
#     print("USER QUESTION")
#     print("==============================================")
#     print(question)

#     config = {"configurable": {"thread_id": thread_id}}

#     existing_state = graph.get_state(config)
#     messages = []
#     if not existing_state.values.get("messages"):
#         messages.append({
#             "role": "system",
#             "content": (
#                 "Answer using the tool results you already have. "
#                 "Do not call the same tool more than once with a similar query, and do not call "
#                 "both tools for the same sub-question if one tool already returned a clear answer. "
#                 "Use search_documents ONLY for AcmeNova company facts (head office, employees, policies). "
#                 "Use search_database for questions about users, accounts, emails, or signup data. "
#                 "Use web_search for any real-world person's name (e.g. Vijay) that is not an AcmeNova "
#                 "employee listed in search_documents results — do not call search_documents again for that name."
#             )
#         })
#     messages.append({"role": "user", "content": question})

#     try:
#         result = graph.invoke({"query": question, "messages": messages}, config=config)
#         final_message = result["messages"][-1]
#         answer = final_message.content

#     except RateLimitError as exc:
#         answer = (
#             "The AI service has hit its rate/token limit for now. "
#             f"Please try again shortly.\n\nDetails: {exc}"
#         )
#     except APIConnectionError as exc:
#         answer = f"Could not reach the AI service (network issue). Details: {exc}"
#     except APIStatusError as exc:
#         answer = f"The AI service returned an error (status {exc.status_code}). Details: {exc}"
#     except Exception as exc:
#         answer = f"Unexpected error while generating the answer: {exc}"

#     print("\n")
#     print("==============================================")
#     print("FINAL ANSWER")
#     print("==============================================")
#     print(answer)

#     return answer


# if __name__ == "__main__":
#     question = "latest ipl winner 2026 and the AcmeNova head office location and \"How many inactive users?\""
#     # question = "How many users are there?"
#     thread_id="demo"
#     ask_question(question, thread_id=thread_id)