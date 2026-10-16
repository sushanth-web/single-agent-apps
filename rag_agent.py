from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents import create_agent
import streamlit as st
import os


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def document_processor(path):

    loader = PyPDFDirectoryLoader(path)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splitted_text = splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

    llm = ChatGroq(model="openai/gpt-oss-20b")

    vector_store = InMemoryVectorStore.from_documents(
        documents = splitted_text,
        embedding = embeddings
    )

    @tool
    def retrieve_context(query:str):
        """ 
            You are a helpful agent that will provide me the answers about the document based on the user query
        """
        docs = vector_store.similarity_search(query)
        context=""
        for doc in docs:
            context += doc.page_content + "\n\n"
        return context

    agent = create_agent(
        model = llm,
        tools = [retrieve_context],
        system_prompt=""""
            You are a document question-answering agent.

            You have access to a tool called `retrieve_context`.

            IMPORTANT RULES:
            1. For EVERY user question about the uploaded documents, you MUST call
            `retrieve_context` BEFORE answering.
            2. NEVER answer document-related questions from your own knowledge.
            3. If the retrieved context contains the answer, answer using that context.
            4. If the retrieved context does not contain the answer, say that the
            information could not be found in the uploaded documents.
            5. Do not ask the user which document they mean. Search the knowledge base first.
            """,
        checkpointer=InMemorySaver()
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True
# pdf ui
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="select files", type = {"pdf"}, accept_multiple_files = True)
    if uploaded:
        with st.spinner("Processing..."):
            path = "./doc_files"
            for file in uploaded:
                file_path = os.path.join(path, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue() )
            
            document_processor(path)
            st.rerun()


if st.session_state.document_uploaded and st.session_state.agent:

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    query = st.chat_input("Ask about the docs...")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        
        response = st.session_state.agent.invoke(
                {"messages":[{"role":"user","content":query}]},
                {"configurable":{"thread_id":"1"}}
                )
        
        with st.spinner("Processing..."):
            answer = response["messages"][-1].content
            st.session_state.messages.append({"role":"ai","content":answer})
            st.chat_message("ai").markdown(answer)
