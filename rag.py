import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- GLOBAL DBs ----------------
BASE_DB = None
USER_DB = None

# ---------------- EMBEDDINGS ----------------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------- TEXT SPLITTING ----------------
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,        # 🔥 smaller = better accuracy
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

# ---------------- LOAD BASE KNOWLEDGE ----------------
def load_base_knowledge():
    global BASE_DB

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    sources = [
        os.path.join(BASE_DIR, "data", "sample.pdf"),
        os.path.join(BASE_DIR, "data", "knowledge.txt"),
    ]

    all_docs = []

    for path in sources:
        if os.path.exists(path):
            try:
                if path.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                else:
                    loader = TextLoader(path, encoding="utf-8")

                docs = loader.load()
                all_docs.extend(docs)

            except Exception as e:
                print(f"❌ Error loading {path}: {e}")

    if not all_docs:
        print("⚠️ No base documents found.")
        return

    splits = split_docs(all_docs)

    BASE_DB = FAISS.from_documents(splits, get_embeddings())

    print(f"✅ Base knowledge loaded: {len(splits)} chunks")

# ---------------- ADD USER PDF ----------------
def add_user_pdf(path):
    global USER_DB

    try:
        loader = PyPDFLoader(path)
        docs = loader.load()

        splits = split_docs(docs)

        new_db = FAISS.from_documents(splits, get_embeddings())

        if USER_DB:
            USER_DB.merge_from(new_db)
        else:
            USER_DB = new_db

        print(f"✅ User PDF added: {len(splits)} chunks")

    except Exception as e:
        print(f"❌ Failed to process PDF: {e}")

# ---------------- GET RETRIEVER ----------------
def get_retriever():
    global BASE_DB, USER_DB

    if BASE_DB and USER_DB:
        merged = BASE_DB
        merged.merge_from(USER_DB)

        return merged.as_retriever(
            search_type="mmr",              # 🔥 better than similarity
            search_kwargs={"k": 3}
        )

    if BASE_DB:
        return BASE_DB.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )

    if USER_DB:
        return USER_DB.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )

    return None

# ---------------- DEBUG FUNCTION ----------------
def debug_retrieval(docs):
    print("\n🔎 --- RETRIEVED CHUNKS ---\n")
    for i, d in enumerate(docs):
        print(f"Chunk {i+1}:\n{d.page_content[:200]}\n")