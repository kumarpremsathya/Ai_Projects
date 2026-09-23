# import streamlit as st
# from langraph_with_rag_tool import ask_question

# st.set_page_config(page_title="AcmeNova RAG Assistant", page_icon="🤖")
# st.title("AcmeNova RAG + Web Search Assistant")

# st.write(
#     "Ask about AcmeNova company info (from the knowledge base) "
#     "or anything else (answered via web search)."
# )

# question = st.text_input("Your question", value=" ")

# if st.button("Ask") and question.strip():
#     with st.spinner("Thinking..."):
#         answer = ask_question(question)
#     st.subheader("Answer")
#     st.write(answer)


import uuid
import streamlit as st
from langraph_with_multiple_tool import ask_question

st.set_page_config(page_title="AcmeNova RAG Assistant", page_icon="🤖")
st.title("LangraphAssistant")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

if st.sidebar.button("+ New Chat"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []

for role, content in st.session_state.history:
    st.chat_message(role).write(content)

question = st.chat_input("Ask a question...")

if question:
    st.chat_message("user").write(question)
    with st.spinner("Thinking..."):
        try:
            answer = ask_question(question, thread_id=st.session_state.thread_id)
        except Exception as exc:
            answer = f"Something went wrong: {exc}"
    st.chat_message("assistant").write(answer)
    st.session_state.history.append(("user", question))
    st.session_state.history.append(("assistant", answer))