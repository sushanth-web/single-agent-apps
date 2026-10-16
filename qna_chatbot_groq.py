from langchain_groq import ChatGroq
import streamlit as st
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import GoogleSerperAPIWrapper

llm = ChatGroq(model="openai/gpt-oss-20b", streaming = True)

search = GoogleSerperAPIWrapper()

agent = create_agent(
    model = llm,
    tools = [search.run],
    system_prompt = "You are a light weight chatbot which is implemented for its quick responses",
    checkpointer = MemorySaver()
)

st.title("Groq QnA Chatbot")
st.markdown("Speed first ai chatbot made using groq and gpt-oss-20b")

query = st.chat_input("Ask Anything...")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if query:
    st.session_state.messages.append({
            "role": "user",
            "content": query
        })
    
    st.chat_message("user").markdown(query)
    
    response = agent.stream(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":"1"}},
        stream_mode = "messages"
        )
    
    with st.chat_message("ai"):
        message_placeholder = st.empty()
        full_message = ""
        for chunk,meta_data in response:
            full_message+=chunk.content
            message_placeholder.markdown(full_message)
    st.session_state.messages.append({
                "role": "assistant",
                "content": full_message
            })