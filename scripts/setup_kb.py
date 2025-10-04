import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- LOAD API KEY ---
load_dotenv() 

# --- 1. DEFINE PATHS ---
DATA_PATH = "slp_knowledge_base/" 
DB_PATH = "slp_vector_db/"

# --- 2. LOAD THE DOCUMENTS ---
print("Loading documents...")

# Load PDF files
pdf_loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.pdf",    # Pattern to load only PDF files
    loader_cls=PyPDFLoader, # Specify the loader class for PDFs
    use_multithreading=True
)
pdf_documents = pdf_loader.load()

# Load DOCX files
docx_loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.docx",   # Pattern to load only DOCX files
    loader_cls=Docx2txtLoader, # Specify the loader class for DOCX
    use_multithreading=True
)
docx_documents = docx_loader.load()

# Combine the loaded documents into one list
documents = pdf_documents + docx_documents

if not documents:
    print("No documents found. Please check your DATA_PATH and make sure you have documents in the folder.")
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
    print(f"✅ Success! Your vector database using OpenAI is ready in the '{DB_PATH}' folder.")
    print("--------------------------------------------------")