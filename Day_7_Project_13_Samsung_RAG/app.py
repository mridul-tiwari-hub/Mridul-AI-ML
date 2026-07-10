# -*- coding: utf-8 -*-
# --- SQLite Override for ChromaDB on Streamlit Cloud ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
    pass

import streamlit as st
import os
import tempfile
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- Page Configuration ---
st.set_page_config(
    page_title="Samsung Manual Assistant",
    page_icon="💧",
    layout="wide"
)

# Custom premium styling
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
</style>
<div class="header-box">
    <h2>🤖 Project 13: Samsung Washing Machine Assistant</h2>
    <p>A Retrieval-Augmented Generation (RAG) chatbot trained on the official Samsung Washing Machine manual. Ask questions about error codes, cycles, and settings.</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Setup ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Auto-load API key if set in environment or secrets
    env_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API Key",
        value=env_key,
        type="password",
        help="Enter your secret key starting with 'sk-'"
    )
    st.markdown("---")
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload the Samsung HTML Manual", type=["html"])
    st.markdown("*(If no file is uploaded, the app will use the default `model.html` manual in the project folder.)*")
    
    # Clear chat
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- RAG Pipeline (Cached for Performance) ---
@st.cache_resource(show_spinner=False)
def setup_rag_pipeline(api_key, file_path):
    # Set the API key
    os.environ["OPENAI_API_KEY"] = api_key

    # Initialize models
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load and process document using BSHTMLLoader
    loader = BSHTMLLoader(file_path=file_path)
    machine_docs = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(machine_docs)

    # Create vector database
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Define prompt
    prompt = ChatPromptTemplate.from_template("""You are a senior support assistant for Samsung Washing Machine manuals.
    Use the following pieces of retrieved context from the manual to answer the user's question.
    If you don't know the answer or if the context doesn't mention it, reply "I don't know based on the provided documentation."
    Do not make up facts. Keep your answer concise and direct (maximum three sentences).

    Question: {question}
    Context: {context}
    Answer:""")

    # Build the chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain

# --- Chat State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if user_input := st.chat_input("E.g., What does the 4C error code mean?"):

    # 1. Validate API Key
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    # 2. Determine File Path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(base_dir, "model.html")
    
    file_path = default_path
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
    elif not os.path.exists(file_path):
        st.error("❌ Default manual `model.html` not found. Please upload the manual file in the sidebar.")
        st.stop()

    # 3. Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 4. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching washing machine manual..."):
            try:
                # Load pipeline and get response
                rag_chain = setup_rag_pipeline(api_key, file_path)
                response = rag_chain.invoke(user_input).content

                # Display and save response
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred while calling the model: {str(e)}")
