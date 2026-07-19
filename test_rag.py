"""
Backend Test Script - DRDO AI Repository

This script tests the core RAG pipeline WITHOUT any frontend:
1. Load a local PDF
2. Split into chunks
3. Create embeddings
4. Store in FAISS
5. Ask questions in terminal and get answers from Mistral (Ollama)
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document
import PyPDF2
import os

# ====== CONFIG ======
PDF_PATH = "rldl_notes.pdf"  # put your PDF file name here
MAX_PAGES = 50               # limit for faster testing
# =====================

def load_pdf(path: str, max_pages: int = None) -> Document:
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    reader = PyPDF2.PdfReader(path)
    total_pages = len(reader.pages)
    if max_pages is None:
        max_pages = total_pages
    max_pages = min(max_pages, total_pages)

    print(f"[INFO] Loading '{path}' ({total_pages} pages), using first {max_pages} pages")
    text = ""
    for i in range(max_pages):
        page_text = reader.pages[i].extract_text()
        if page_text:
            text += page_text + "\n"

    return Document(
        page_content=text,
        metadata={"source": os.path.basename(path), "total_pages": total_pages, "used_pages": max_pages},
    )

def build_vectorstore(doc: Document):
    print("[INFO] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents([doc])
    print(f"[INFO] Created {len(chunks)} chunks")

    print("[INFO] Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("[INFO] Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("[INFO] Vectorstore ready")
    return vectorstore

def main():
    # 1) Load PDF
    doc = load_pdf(PDF_PATH, MAX_PAGES)

    # 2) Build vectorstore
    vectorstore = build_vectorstore(doc)

    # 3) Setup LLM
    print("[INFO] Connecting to Mistral (Ollama)...")
    llm = Ollama(model="mistral", temperature=0.3)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # 4) Interactive Q&A
    print("\n=== DRDO AI Repository - Backend Test ===")
    print("PDF:", PDF_PATH)
    print("Type your question (or 'quit' to exit)")

    while True:
        q = input("\nQuestion: ")
        if q.lower() in ["quit", "exit"]:
            break

        print("[INFO] Searching and generating answer...")
        result = qa.invoke({"query": q})
        print("\nAnswer:")
        print(result["result"])

        print("\n[Sources used]")
        for i, d in enumerate(result["source_documents"], 1):
            print(f"\nSource {i} ({d.metadata.get('source', '')}):")
            print(d.page_content[:400].replace("\n", " ") + "...")
            print("-" * 60)

    print("\n[INFO] Test session ended.")

if __name__ == "__main__":
    main()
