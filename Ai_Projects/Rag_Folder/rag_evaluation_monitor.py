import json
import os
import time
from datetime import datetime, timezone
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree


# ============================================================
# 1. LOAD API KEYS AND LANGSMITH CONFIGURATION
# ============================================================

load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

if not GROK_API_KEY:
    raise ValueError("GROK_API_KEY is missing in the .env file")

if not LANGSMITH_API_KEY:
    raise ValueError("LANGSMITH_API_KEY is missing in the .env file")

os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"

# LangSmith project name
os.environ["LANGSMITH_PROJECT"] = "simple-rag-groundedness-project"


# ============================================================
# 2. LOAD DOCUMENTS
# ============================================================

print("Loading documents...")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

documents_from_urls = []

for url in urls:
    loader = WebBaseLoader(url)
    loaded_documents = loader.load()
    documents_from_urls.extend(loaded_documents)

print(f"Number of original documents: {len(documents_from_urls)}")


# ============================================================
# 3. SPLIT DOCUMENTS INTO SMALL CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

document_chunks = text_splitter.split_documents(
    documents_from_urls
)

print(f"Number of document chunks: {len(document_chunks)}")


# ============================================================
# 4. CREATE VECTOR STORE
# ============================================================

print("Creating vector store...")

# Local embeddings, since Groq does not provide an embeddings API
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = InMemoryVectorStore.from_documents(
    documents=document_chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# ============================================================
# 5. CREATE THE RAG LLM
# ============================================================

rag_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROK_API_KEY
)

APPLICATION_NAME = "simple-rag-groundedness-app"
MODEL_NAME = "openai/gpt-oss-120b"
METRICS_LOG_FILE = "metrics_log.jsonl"


def log_request_metrics(metrics: dict) -> None:
    """Append one request's metrics as a JSON line for later analysis."""
    with open(METRICS_LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(metrics) + "\n")


# ============================================================
# 6. CREATE TRACEABLE RAG APPLICATION
# ============================================================

@traceable(name="simple_rag_application")
def rag_bot(question: str) -> dict:
    """
    Executes the RAG application.

    Returns:
        {
            "answer": generated answer,
            "documents": retrieved documents
        }
    """

    print(f"\nProcessing question: {question}")

    request_timestamp = datetime.now(timezone.utc).isoformat()
    start_time = time.time()
    status = "Success"
    error_message = None
    input_tokens = output_tokens = total_tokens = None
    answer = None
    retrieved_documents = []

    try:
        # Step 1: Retrieve relevant documents
        retrieved_documents = retriever.invoke(question)

        # Step 2: Convert documents into context text
        context = "\n\n".join(document.page_content for document in retrieved_documents)

        # Step 3: Create the prompt
        system_prompt = f"""
            You are a helpful question-answering assistant.

            Answer the user's question using ONLY the information
            provided in the context below.

            Rules:
            1. Do not use outside knowledge.
            2. Do not invent information.
            3. If the answer is not available in the context,
            say "I don't know."
            4. Keep the answer short and clear.

            Context:
            {context}
            """

        # Step 4: Generate the answer
        response = rag_llm.invoke(
            [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        answer = response.content

        # Token usage, when the model/provider reports it
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")

        return {
            "answer": answer,
            "documents": retrieved_documents
        }

    except Exception as exc:
        status = "Failure"
        error_message = str(exc)
        raise

    finally:
        response_time_seconds = round(time.time() - start_time, 3)
        run_tree = get_current_run_tree()
        run_id = str(run_tree.id) if run_tree else None

        # Serialize retrieved documents so they can be stored as JSON
        retrieved_documents_log = [
            {
                "content": document.page_content,
                "metadata": document.metadata,
            }
            for document in retrieved_documents
        ]

        log_request_metrics({
            "application_name": APPLICATION_NAME,
            "model_name": MODEL_NAME,
            "request_id": run_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "response_time_seconds": response_time_seconds,
            "status": status,
            "error_message": error_message,
            "request_timestamp": request_timestamp,
            "langsmith_run_id": run_id,
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_documents_log,
            "num_retrieved_documents": len(retrieved_documents_log),
        })


# ============================================================
# 7. DEFINE THE GROUNDEDNESS OUTPUT FORMAT
# ============================================================

class GroundednessGrade(TypedDict):
    explanation: Annotated[
        str,
        "Explain why the answer is or is not grounded in the documents"
    ]

    grounded: Annotated[
        bool,
        "True if the answer is supported by the documents, otherwise False"
    ]


class RelevanceGrade(TypedDict):
    explanation: Annotated[
        str,
        "Explain why the content is or is not relevant to the question"
    ]

    relevant: Annotated[
        bool,
        "True if the content is relevant to the question, otherwise False"
    ]


# ============================================================
# 8. CREATE THE GROUNDEDNESS JUDGE
# ============================================================

groundedness_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROK_API_KEY
).with_structured_output(
    GroundednessGrade
)


groundedness_instructions = """
    You are an evaluator for a Retrieval-Augmented Generation application.

    You will receive:

    1. A QUESTION
    2. RETRIEVED DOCUMENTS
    3. A GENERATED ANSWER

    Your task is to determine whether the generated answer is grounded
    in the retrieved documents.

    Evaluation rules:

    1. Every factual claim in the answer must be supported by the documents.
    2. The answer must not contain information from outside the documents.
    3. If the answer contains an unsupported fact, mark grounded as False.
    4. If the answer says "I don't know" because the documents do not
    contain the answer, mark grounded as True.
    5. Do not judge whether the documents themselves are absolutely true.
    6. Judge only whether the answer is supported by the documents.

    Return:

    - explanation: a short explanation
    - grounded: True or False
    """


relevance_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROK_API_KEY
).with_structured_output(
    RelevanceGrade
)


answer_relevance_instructions = """
    You are evaluating a generated answer for relevance.

    You will receive a question and an answer. Mark relevant as True only when
    the answer directly addresses the question and is helpful and concise.
    Return an explanation and a boolean score.
    """


retrieval_relevance_instructions = """
    You are evaluating retrieved documents for relevance.

    You will receive a question and retrieved documents. Mark relevant as True
    when the documents contain information that can help answer the question.
    Some unrelated information is acceptable. Return an explanation and a
    boolean score.
    """


# ============================================================
# 9. DEFINE THE GROUNDEDNESS EVALUATOR
# ============================================================

def groundedness_evaluator(inputs: dict,outputs: dict) -> dict:
    """
    Checks whether the generated RAG answer is supported
    by the retrieved documents.
    """

    question = inputs["question"]
    generated_answer = outputs["answer"]
    retrieved_documents = outputs["documents"]

    # Combine all retrieved document chunks
    document_text = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )

    evaluator_input = f"""
        QUESTION:
        {question}

        RETRIEVED DOCUMENTS:
        {document_text}

        GENERATED ANSWER:
        {generated_answer}
        """

    # Ask the judge LLM to evaluate groundedness
    grade = groundedness_llm.invoke(
        [
            {
                "role": "system",
                "content": groundedness_instructions
            },
            {
                "role": "user",
                "content": evaluator_input
            }
        ]
    )

    print("\nGroundedness evaluation:")
    print("Explanation:", grade["explanation"])
    print("Grounded:", grade["grounded"])

    # Return a LangSmith-compatible evaluation result
    return {
        "key": "groundedness",
        "score": grade["grounded"],
        "comment": grade["explanation"]
    }


def answer_relevance_evaluator(inputs: dict, outputs: dict) -> dict:
    """Checks whether the generated answer addresses the question."""
    evaluator_input = f"""
        QUESTION:
        {inputs["question"]}

        GENERATED ANSWER:
        {outputs["answer"]}
        """

    grade = relevance_llm.invoke(
        [
            {"role": "system", "content": answer_relevance_instructions},
            {"role": "user", "content": evaluator_input}
        ]
    )

    print("\nAnswer relevance evaluation:")
    print("Explanation:", grade["explanation"])
    print("Relevant:", grade["relevant"])

    return {
        "key": "answer_relevance",
        "score": grade["relevant"],
        "comment": grade["explanation"]
    }


def retrieval_relevance_evaluator(inputs: dict, outputs: dict) -> dict:
    """Checks whether the retrieved documents are relevant to the question."""
    document_text = "\n\n".join(
        document.page_content
        for document in outputs["documents"]
    )
    evaluator_input = f"""
        QUESTION:
        {inputs["question"]}

        RETRIEVED DOCUMENTS:
        {document_text}
        """

    grade = relevance_llm.invoke(
        [
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": evaluator_input}
        ]
    )

    print("\nRetrieval relevance evaluation:")
    print("Explanation:", grade["explanation"])
    print("Relevant:", grade["relevant"])

    return {
        "key": "retrieval_relevance",
        "score": grade["relevant"],
        "comment": grade["explanation"]
    }


# ============================================================
# 10. TEST THE RAG APPLICATION ON ONE QUESTION
# ============================================================

if __name__ == "__main__":

    test_question = "What are the main components of an LLM-powered agent?"

    test_result = rag_bot(test_question)

    print("\n==============================")
    print("RAG ANSWER")
    print("==============================")
    print(test_result["answer"])

    print("\nNumber of retrieved documents:")
    print(len(test_result["documents"]))

    real_time_evaluators = [
        groundedness_evaluator,
        answer_relevance_evaluator,
        retrieval_relevance_evaluator
    ]

    for evaluator in real_time_evaluators:
        result = evaluator(
            inputs={"question": test_question},
            outputs=test_result,
        )
        print(f"\nReal-time {result['key']} score:", result["score"])

    print("\nReal-time evaluations complete.")

    # ========================================================
    # 11. CREATE LANGSMITH DATASET
    # ========================================================

    print("\nCreating LangSmith dataset...")

    langsmith_client = Client()

    dataset_name = "simple-rag-groundedness-dataset"

    # Reuse the dataset if it already exists, otherwise create it
    if langsmith_client.has_dataset(dataset_name=dataset_name):
        dataset = langsmith_client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = langsmith_client.create_dataset(
            dataset_name=dataset_name,
            description="Dataset for evaluating RAG answer groundedness"
        )

    # Test questions
    examples = [
        {
            "question": "What are the main components of an LLM-powered agent?"
        },
        {
            "question": "What is the role of memory in an LLM-powered agent?"
        },
        {
            "question": "What is task decomposition?"
        }
    ]

    # Only seed examples if the dataset is empty, to avoid duplicates on rerun
    existing_examples = list(langsmith_client.list_examples(dataset_id=dataset.id))
    if not existing_examples:
        for example in examples:
            langsmith_client.create_example(
                inputs={
                    "question": example["question"]
                },
                dataset_id=dataset.id
            )

    print("Dataset created successfully.")


    # ========================================================
    # 12. DEFINE THE LANGSMITH TARGET FUNCTION
    # ========================================================

    def rag_target(inputs: dict) -> dict:
        """
        LangSmith calls this function for every dataset example.
        """

        question = inputs["question"]

        result = rag_bot(question)

        return {
            "answer": result["answer"],
            "documents": result["documents"]
        }


    # ========================================================
    # 13. RUN RAG EVALUATIONS
    # ========================================================

    print("\nStarting LangSmith RAG evaluations...")

    evaluation_results = langsmith_client.evaluate(
        rag_target,
        data=dataset_name,
        evaluators=[
            groundedness_evaluator,
            answer_relevance_evaluator,
            retrieval_relevance_evaluator
        ],
        experiment_prefix="simple-rag-groundedness"
    )

    print("\nEvaluation completed.")

    print(
        "Open your LangSmith project to view:"
        " traces, retrieved documents, answers, and groundedness scores."
    )