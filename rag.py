from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

vector_store = None


def load_base_knowledge():
    global vector_store

    loader = PyPDFLoader("data/knowledge.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = FakeEmbeddings(size=1536)

    vector_store = FAISS.from_documents(chunks, embeddings)


def add_user_pdf(path):
    global vector_store

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()

    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        vector_store.add_documents(chunks)


def get_retriever():
    if vector_store:
        return vector_store.as_retriever(search_kwargs={"k": 4})
    return None