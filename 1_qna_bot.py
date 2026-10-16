from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

st.title("ASKbuddy")
st.markdown("QnA Bot made using LangChain and Gemini AI")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Get user input
query = st.chat_input("Ask Anything...")

if query:

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.chat_message("user").markdown(query)

    # Get Gemini response
    res = llm.invoke(query)

    # Get only the text
    answer = res.text

    # Add AI response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.chat_message("assistant").markdown(answer)