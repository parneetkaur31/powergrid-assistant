from dotenv import load_dotenv
import os

load_dotenv()

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from rank_bm25 import BM25Okapi
from langchain.schema import Document

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

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = vectorstore.similarity_search("", k=100)
    bm25, texts = build_bm25(docs)

    return vector_retriever, bm25, texts, docs


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


def generate_hypothetical_answer(query):

    llm = ChatOpenAI(temperature=0)

    prompt = f"""
    Write a short technical explanation that answers the question.

    Question: {query}

    Hypothetical answer:
    """

    response = llm.invoke(prompt)

    return response.content


def ask_question(query):

    hyde_query = generate_hypothetical_answer(query)
    docs = hybrid_search(hyde_query)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer the question using the following context.

    Context:
    {context}

    Question: {query}
    """

    llm = ChatOpenAI(temperature=0)

    answer = llm.invoke(prompt)

    sources = set()

    for doc in docs:
        if "source" in doc.metadata:
            sources.add(os.path.basename(doc.metadata["source"]))

    return answer.content, sources


def build_bm25(docs):
    texts = [doc.page_content for doc in docs]
    tokenized = [text.split() for text in texts]

    bm25 = BM25Okapi(tokenized)

    return bm25, texts


def hybrid_search(query):

    vector_retriever, bm25, texts, docs = load_retriever()

    vector_docs = vector_retriever.get_relevant_documents(query)

    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:3]

    bm25_docs = [docs[i] for i in top_indices]

    return vector_docs + bm25_docs