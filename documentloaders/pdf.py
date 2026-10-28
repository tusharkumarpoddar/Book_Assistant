
from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader(
    "documentloaders/sample.pdf",
   
)

docs = data.load()

print(docs)