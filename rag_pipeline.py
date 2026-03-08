from dotenv import load_dotenv
import os

load_dotenv()

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

VECTOR_PATH = "vectorstore"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def load_retriever():
    vectorstore = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever


def build_chain():
    retriever = load_retriever()

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa


def ask_question(query):

    qa = build_chain()

    result = qa({"query": query})

    answer = result["result"]

    sources = set()

    for doc in result["source_documents"]:
        sources.add(doc.metadata.get("source", "document"))

    return answer, sources