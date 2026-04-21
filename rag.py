import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

default_db = None

# ---------------- LOAD DEFAULT ----------------
def load_default_doc():
    global default_db

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(BASE_DIR, "data", "sample.pdf")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    default_db = FAISS.from_documents(splits, embeddings)

# ---------------- ADD USER PDF ----------------
def add_user_pdf(path):
    global default_db

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    new_db = FAISS.from_documents(splits, embeddings)

    if default_db:
        default_db.merge_from(new_db)
    else:
        default_db = new_db

# ---------------- GET RETRIEVER ----------------
def get_retriever():
    global default_db
    if default_db:
        return default_db.as_retriever()
    return None