import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

DOCS_PATH = "docs"
VECTOR_PATH = "vectorstore"
os.makedirs(VECTOR_PATH, exist_ok=True)

# embedding model (fast + good for technical docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vector database
vectorstore = FAISS.from_texts(["init"], embeddings)

# parent store
store = InMemoryStore()

# parent chunks (context chunks)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
)

# child chunks (search chunks)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=100
)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

documents = []

for file in os.listdir(DOCS_PATH):
    if file.endswith(".pdf"):
        path = os.path.join(DOCS_PATH, file)
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)

print("Loaded documents:", len(documents))

retriever.add_documents(documents)

print("Creating vector index...")

vectorstore.save_local(VECTOR_PATH)

print("Vectorstore saved.")