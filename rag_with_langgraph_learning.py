import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

PDF_PATH = "handbook.pdf"
VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "handbook"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
APPLICATION_NAME = "langgraph-pdf-rag"
METRICS_LOG_FILE = "metrics_log.jsonl"

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing in the .env file")

if not os.getenv("LANGSMITH_API_KEY"):
    raise ValueError("LANGSMITH_API_KEY is missing in the .env file")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langgraph-pdf-rag"


class RAGState(TypedDict):
    question: str
    query: str
    docs: List[Document]
    answer: str
    grade: str
    retries: int


def log_request_metrics(metrics: dict) -> None:
    with open(METRICS_LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(metrics) + "\n")


embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
reranker = CrossEncoder(RERANK_MODEL)


def load_pdf(pdf_path: str) -> List[Document]:
    if not Path(pdf_path).exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            f"Put your PDF in the same folder as this script "
            f"or change PDF_PATH."
        )

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = doc.metadata.get("source", pdf_path)

    print(f"[LOAD] Pages loaded: {len(docs)}")

    if docs:
        print("\n[LOAD] First page sample:")
        print(docs[0].page_content[:500])
        print()

    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"[CHUNK] Number of chunks: {len(chunks)}")
    return chunks


def create_vector_store(chunks: List[Document]) -> Chroma:
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH,
    )
    print(f"[VECTOR DB] Created at: {VECTOR_DB_PATH}")
    return store


def open_vector_store() -> Chroma:
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )
    print(f"[VECTOR DB] Opened: {VECTOR_DB_PATH}")
    return store


def create_retriever(store: Chroma):
    return store.as_retriever(search_type="mmr", search_kwargs={"k": 12, "fetch_k": 40})


def make_retrieve_node(retriever):
    def retrieve(state: RAGState) -> dict:
        q = state.get("query") or state["question"]
        print("\n[RETRIEVE]")
        print("Query:", q)
        docs = retriever.invoke(q)
        print("Retrieved documents:", len(docs))
        return {"docs": docs, "query": q}

    return retrieve


def rerank(state: RAGState) -> dict:
    docs = state["docs"]
    if not docs:
        return {"docs": []}

    pairs = [(state["query"], doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: -x[0])
    top_docs = [doc for _, doc in ranked[:4]]

    print("\n[RERANK]")
    print("Input documents:", len(docs))
    print("Documents kept:", len(top_docs))
    return {"docs": top_docs}


def format_context(docs: List[Document]) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        display_page = page + 1 if isinstance(page, int) else page
        formatted.append(f"[{source} p{display_page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


SYSTEM_PROMPT = """
You answer only from the context below.

If the answer is not in the context, say:
"I could not find this in the documents."

Do not invent information.

Cite the source file and page for every factual claim.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
chain = prompt | llm


def generate(state: RAGState) -> dict:
    context = format_context(state["docs"])
    print("\n[GENERATE]")
    response = chain.invoke({"context": context, "question": state["question"]})
    answer = response.content
    return {"answer": answer}


class Grade(BaseModel):
    verdict: str = Field(description="useful or not_useful")


grader = llm.with_structured_output(Grade)


def grade_docs(state: RAGState) -> dict:
    context = format_context(state["docs"])
    print("\n[GRADE]")
    result = grader.invoke(
        f"""
Question:
{state["question"]}

Context:
{context}

Can this context answer the question?

Return:
useful
or
not_useful
"""
    )
    verdict = result.verdict.strip().lower()
    if verdict not in {"useful", "not_useful"}:
        verdict = "not_useful"
    print("Grade:", verdict)
    return {"grade": verdict}


def rewrite(state: RAGState) -> dict:
    print("\n[REWRITE]")
    response = llm.invoke(
        f"""
Rewrite the following question into a better
query for document search.

Keep the original meaning.
Make the query precise and useful for semantic search.

Question:
{state["question"]}
"""
    )
    better_query = response.content.strip()
    retries = state.get("retries", 0) + 1
    print("New query:", better_query)
    print("Retry number:", retries)
    return {"query": better_query, "retries": retries}


def route(state: RAGState) -> str:
    if state["grade"] == "useful" or state.get("retries", 0) >= 2:
        return "generate"
    return "rewrite"


def build_graph(retriever):
    retrieve = make_retrieve_node(retriever)
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("grade_docs", grade_docs)
    builder.add_node("rewrite", rewrite)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "grade_docs")
    builder.add_conditional_edges("grade_docs", route, {"generate": "generate", "rewrite": "rewrite"})
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph


def ingest_pdf():
    print("\n" + "=" * 60)
    print("STARTING INGESTION")
    print("=" * 60)
    docs = load_pdf(PDF_PATH)
    chunks = chunk_documents(docs)
    store = create_vector_store(chunks)
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    return store


@traceable(name="langgraph_pdf_rag")
def answer_question(graph, question: str, thread_id: str = "learning-user-1"):
    initial_state: RAGState = {
        "question": question,
        "query": "",
        "docs": [],
        "answer": "",
        "grade": "",
        "retries": 0,
    }
    config = {"configurable": {"thread_id": thread_id}}
    request_timestamp = datetime.now(timezone.utc).isoformat()
    start_time = time.time()
    status = "Success"
    error_message = None
    result = None

    print("\n" + "=" * 60)
    print("QUESTION")
    print(question)
    print("=" * 60)
    try:
        result = graph.invoke(initial_state, config)
        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(result["answer"])
        print("=" * 60)
        return result
    except Exception as exc:
        status = "Failure"
        error_message = str(exc)
        raise
    finally:
        run_tree = get_current_run_tree()
        run_id = str(run_tree.id) if run_tree else None
        documents = result["docs"] if result else []
        log_request_metrics({
            "application_name": APPLICATION_NAME,
            "model_name": LLM_MODEL,
            "request_id": run_id,
            "langsmith_run_id": run_id,
            "thread_id": thread_id,
            "question": question,
            "answer": result["answer"] if result else None,
            "grade": result["grade"] if result else None,
            "retries": result.get("retries") if result else None,
            "retrieved_documents": [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents],
            "num_retrieved_documents": len(documents),
            "response_time_seconds": round(time.time() - start_time, 3),
            "status": status,
            "error_message": error_message,
            "request_timestamp": request_timestamp,
        })


def stream_question(graph, question: str, thread_id: str = "learning-user-1"):
    initial_state: RAGState = {
        "question": question,
        "query": "",
        "docs": [],
        "answer": "",
        "grade": "",
        "retries": 0,
    }
    config = {"configurable": {"thread_id": thread_id}}
    print("\n" + "=" * 60)
    print("STREAMING GRAPH")
    print("=" * 60)
    for step in graph.stream(initial_state, config):
        print(step)


def evaluate_pipeline(graph, testset):
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset

    rows = []
    for question, truth in testset:
        result = graph.invoke({
            "question": question,
            "query": "",
            "docs": [],
            "answer": "",
            "grade": "",
            "retries": 0,
        })
        rows.append({
            "question": question,
            "answer": result["answer"],
            "contexts": [doc.page_content for doc in result["docs"]],
            "ground_truth": truth,
        })

    dataset = Dataset.from_list(rows)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print(results)
    return results


def main():
    print("\n")
    print("=" * 70)
    print("RAG + LANGGRAPH LEARNING APPLICATION")
    print("=" * 70)

    if not Path(VECTOR_DB_PATH).exists():
        print("\nVector DB not found.")
        print("Running PDF ingestion...")
        store = ingest_pdf()
    else:
        print("\nExisting Vector DB found.")
        print("Opening existing database...")
        store = open_vector_store()

    retriever = create_retriever(store)
    graph = build_graph(retriever)
    question = input("\nAsk a question about the PDF: ").strip()
    if not question:
        print("No question entered.")
        return
    answer_question(graph, question)


if __name__ == "__main__":
    main()
