import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from retriever import create_qa_chain

st.set_page_config(page_title="Docs RAG", page_icon="📄")
st.title("📄 Personal Docs RAG")
st.caption("Ask anything about your Tesla EDA Project Report")

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return create_qa_chain(vector_store)

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
