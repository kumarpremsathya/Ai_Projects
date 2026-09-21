# import streamlit as st
# from langraph_in_memory import graph
# from langchain_core.messages import HumanMessage

# st.title("Simple LangGraph Chat")

# # thread_id keeps each conversation's memory separate
# thread_id = st.text_input("Thread ID", value="abc123")
# user_input = st.text_input("Your message", value="Add 10 with 20")

# if st.button("Send"):
#     config = {"configurable": {"thread_id": thread_id}}
#     result = graph.invoke({"messages": HumanMessage(content=user_input)}, config=config)

#     st.subheader("Conversation")
#     for msg in result["messages"]:
#         st.write(f"**{msg.type}:** {msg.content}")


import json
import os
import uuid
from datetime import datetime

import streamlit as st
from langraph_in_memory import graph
from langchain_core.messages import HumanMessage

CONVO_FILE = "conversations.json"


def load_conversations():
    if os.path.exists(CONVO_FILE):
        with open(CONVO_FILE, "r") as f:
            return json.load(f)
    return {}


def save_conversations(data):
    with open(CONVO_FILE, "w") as f:
        json.dump(data, f, indent=2)


st.title("Multi-User LangGraph Chat")

convos = load_conversations()

# --- Sidebar: pick user, start new chat, pick past chat ---
with st.sidebar:
    user_id = st.text_input("User ID", value="rahul")
    convos.setdefault(user_id, [])

    if st.button("+ New Chat"):
        new_thread = f"{user_id}_{uuid.uuid4().hex[:8]}"
        convos[user_id].append({
            "thread_id": new_thread,
            "title": f"Chat started {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        })
        save_conversations(convos)
        st.session_state["active_thread"] = new_thread

    st.subheader("Your Conversations")
    for convo in reversed(convos[user_id]):
        if st.button(convo["title"], key=convo["thread_id"]):
            st.session_state["active_thread"] = convo["thread_id"]

# --- Main chat area ---
active_thread = st.session_state.get("active_thread")

if not active_thread:
    st.info("Start a new chat or pick one from the sidebar.")
else:
    config = {"configurable": {"thread_id": active_thread}}

    # show past messages for this thread
    state = graph.get_state(config)
    past_messages = state.values.get("messages", []) if state.values else []

    st.subheader(f"Thread: {active_thread}")
    for msg in past_messages:
        st.write(f"**{msg.type}:** {msg.content}")

    user_input = st.text_input("Your message", key="user_input")
    if st.button("Send") and user_input:
        result = graph.invoke({"messages": HumanMessage(content=user_input)}, config=config)
        st.rerun()