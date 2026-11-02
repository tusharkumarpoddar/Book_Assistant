import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚"
)

st.title("📚 RAG PDF Assistant")
st.write("Upload a PDF and ask questions about its content.")

# Initialize models
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings()

@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b")

embeddings = get_embedding_model()
llm = get_llm()

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""),
    ("human", """Context:
{context}

Question:
{question}
""")
])

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:

    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            try:
                # Load PDF
                loader = PyPDFLoader(temp_path)
                documents = loader.load()

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(documents)

                # Create vector database
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma_db"
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.chat_history = []

                st.success("PDF processed successfully!")

            finally:
                # Delete temporary PDF
                if os.path.exists(temp_path):
                    os.remove(temp_path)

# Chat interface
if st.session_state.vectorstore is not None:

    st.subheader("Ask questions about your PDF")

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    query = st.chat_input("Ask something about your PDF...")

    if query:

        st.session_state.chat_history.append(("user", query))

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )

                docs = retriever.invoke(query)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                final_prompt = prompt.invoke({
                    "context": context,
                    "question": query
                })

                response = llm.invoke(final_prompt)

                st.write(response.content)

                # Display source pages
                pages = sorted(set(
                    doc.metadata.get("page", 0) + 1
                    for doc in docs
                ))

                st.caption(f"Source pages: {', '.join(map(str, pages))}")

                st.session_state.chat_history.append(
                    ("assistant", response.content)
                )

# Reset button
if st.button("Reset"):
    st.session_state.vectorstore = None
    st.session_state.chat_history = []
    st.rerun()
