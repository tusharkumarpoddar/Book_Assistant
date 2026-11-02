import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

pdf_path = input("Enter PDF path: ").strip().strip('"')

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

data = PyPDFLoader(pdf_path)
docs = data.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings()


vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory="chroma_db"
)