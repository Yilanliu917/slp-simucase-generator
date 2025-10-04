import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv() 

# --- 1. DEFINE PATHS (UPDATED) ---
# The paths now point inside the 'data' folder
DATA_PATH = "data/slp_knowledge_base/" 
DB_PATH = "data/slp_vector_db/"

# --- 2. LOAD THE DOCUMENTS ---
print("Loading documents...")

pdf_loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    use_multithreading=True
)
pdf_documents = pdf_loader.load()

docx_loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.docx",
    loader_cls=Docx2txtLoader,
    use_multithreading=True
)
docx_documents = docx_loader.load()

documents = pdf_documents + docx_documents

if not documents:
    print(f"No documents found in {DATA_PATH}. Please check the path.")
else:
    print(f"Loaded {len(documents)} documents.")

    # --- 3. CHUNK THE DOCUMENTS ---
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # --- 4. CREATE EMBEDDINGS ---
    print("Creating embeddings with OpenAI... (This may take a few minutes)")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # --- 5. STORE IN VECTOR DATABASE ---
    print("Storing embeddings in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print("--------------------------------------------------")
    print(f"✅ Success! Your vector database is ready in the '{DB_PATH}' folder.")
    print("--------------------------------------------------")