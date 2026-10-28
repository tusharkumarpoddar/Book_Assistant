
from langchain_community.document_loaders import TextLoader

data = TextLoader(
    "documentloaders/notes.txt",
    encoding="utf-8"
)

docs = data.load()

print(len(docs))
