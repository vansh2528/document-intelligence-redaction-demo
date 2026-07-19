import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from datetime import datetime
import fitz  # PyMuPDF
import io
# --------- Redaction config ----------

import re

REDACT_BLOCK = "████████████"

SENSITIVE_KEYWORDS = [
    "missile", "frequency", "radar", "range", "payload",
    "coordinates", "encryption", "DRDO", "classified"
]

def redact_text(text: str, user_role: str) -> str:
    # Scientist / Admin ko full access
    if user_role.lower() in ["scientist", "admin"]:
        return text

    redacted = text

    # 1) Keywords ko blackout
    for word in SENSITIVE_KEYWORDS:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        redacted = pattern.sub(REDACT_BLOCK, redacted)

    # 2) GPS coordinates jaisa "28.6139° N, 77.2090° E" bhi blackout
    coord_pattern = r"\d{1,2}\.\d+°?\s*[NS],\s*\d{1,3}\.\d+°?\s*[EW]"
    redacted = re.sub(coord_pattern, REDACT_BLOCK, redacted)

    # 3) Site names like "Site-A"
    site_pattern = r"\bSite-[A-Z]\b"
    redacted = re.sub(site_pattern, REDACT_BLOCK, redacted)

    return redacted
# --------- Page config ----------

st.set_page_config(
    page_title="DRDO AI Repository",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🔒 DRDO AI Repository")
st.subheader("Zero-Trace Visual Document Question Answering System")

# Sidebar
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True
    )

    st.markdown("---")
    st.markdown("### 👤 User Role")
    user_role = st.selectbox(
        "Select your clearance level:",
        ["Intern", "Scientist", "Admin"],
        index=0
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    This system answers questions based ONLY on uploaded documents.

    **Key Features:**
    - 100% Offline & Local (Ollama + Mistral)
    - Zero-Trace: PDFs & indexes stay in RAM only
    - Visual Evidence: Highlights exact source paragraph on original page
    - Role-based redaction of sensitive terms
    - Multi-document semantic search
    """)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# Store original PDF bytes in memory per file (for visual evidence)
if 'pdf_store' not in st.session_state:
    st.session_state.pdf_store = {}

# --------- PDF processing ----------

def process_pdfs_in_memory(uploaded_files):
    """
    Load all uploaded PDFs fully in RAM using PyMuPDF,
    extract per-page text with page numbers in metadata,
    and build an in-memory FAISS vector store.
    """
    all_chunks = []

    for uploaded_file in uploaded_files:
        # Read bytes into memory (no disk write)
        pdf_bytes = uploaded_file.read()
        # Save raw bytes in session for later visual highlighting
        st.session_state.pdf_store[uploaded_file.name] = pdf_bytes

        # Open PDF from bytes in memory
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if not text or not text.strip():
                continue

            # Per-page document with metadata
            base_doc = Document(
                page_content=text,
                metadata={
                    "source": uploaded_file.name,
                    "page_num": page_num,
                    "total_pages": doc.page_count
                }
            )

            # Split into semantically sized chunks; metadata is preserved
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=150
            )
            chunks = text_splitter.split_documents([base_doc])
            all_chunks.extend(chunks)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # In-memory FAISS store (no save_local -> zero-trace)
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore, len(all_chunks)

# --------- Visual evidence ----------

def generate_visual_evidence(pdf_bytes, page_num, chunk_text):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num]

    # Text normalize: remove extra whitespace
    def normalize(t):
        return re.sub(r"\s+", " ", t).strip()

    chunk_norm = normalize(chunk_text)
    # 1) Try a middle part (zyada unique hota hai)
    mid_start = max(0, len(chunk_norm) // 4)
    mid_snippet = chunk_norm[mid_start:mid_start + 80]

    # 2) Fallbacks
    snippets = [
        mid_snippet,
        chunk_norm[:80],
        chunk_norm[:50]
    ]

    rects = []
    for snip in snippets:
        if not snip:
            continue
        rects = page.search_for(snip)
        if rects:
            break

    # Agar kuch bhi nahi mila, return None
    if not rects:
        return None

    # Sirf pehla rect le lo (best match)
    r = rects[0]
    page.draw_rect(r, color=(1, 0, 0), width=2)
    page.draw_rect(r, color=(1, 1, 0), fill_opacity=0.3)

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return pix.tobytes("png")

# --------- Document upload processing ----------

if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("Processing documents entirely in RAM..."):
        try:
            vectorstore, total_chunks = process_pdfs_in_memory(uploaded_files)
            st.session_state.vectorstore = vectorstore

            st.success(f"✅ Processed {len(uploaded_files)} documents ({total_chunks} chunks)")
            st.info("🔐 Zero-Trace Mode: No PDFs or indexes are written to disk. All processing is in-memory.")

        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

# --------- Main chat interface ----------

if st.session_state.vectorstore is not None:
    st.markdown("### 💬 Ask Questions")

    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the main objective of this project?"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary")
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

    # Process question
    if ask_button and question:
        with st.spinner("Thinking securely..."):
            try:
                # Setup local LLM via Ollama
                llm = Ollama(model="mistral", temperature=0.3)

                retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 5}
                )
                source_docs = retriever.invoke(question)

            # Redacted copies banao (intern ke liye blackout)
                redacted_docs = []
                for d in source_docs:
                    red_content = redact_text(d.page_content, user_role)
                red_doc = Document(page_content=red_content, metadata=d.metadata)
                redacted_docs.append(red_doc)

            # LLM ko sirf redacted context mile
                context = "\n\n".join([d.page_content for d in redacted_docs])

                prompt = (
                    "You are a DRDO document assistant. "
                    "Use the provided context to answer the user's question accurately. "
                    "If the answer is explicitly found, provide it. "
                    "If the answer is not clearly stated, explain what you know based on the context, or say 'Information not explicitly detailed in the provided documents.' "
                    "Be helpful and precise.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n"
                    "Answer:"
                )

                answer_text = llm.invoke(prompt)

                # Answer ko bhi redaction se guzaro
                safe_answer_text = redact_text(answer_text, user_role)

                # Try to produce visual evidence from the best source doc
                evidence_img = None
                evidence_page = None
                evidence_source = None

                if source_docs:
                    best_doc = source_docs[0]
                    evidence_source = best_doc.metadata.get("source")
                    evidence_page = best_doc.metadata.get("page_num")

                    pdf_bytes = None
                    if evidence_source in st.session_state.pdf_store:
                        pdf_bytes = st.session_state.pdf_store[evidence_source]

                    if pdf_bytes is not None and evidence_page is not None:
                        try:
                            evidence_img = generate_visual_evidence(
                                pdf_bytes,
                                evidence_page,
                                best_doc.page_content
                            )
                        except Exception as e:
                            st.warning(f"Visual evidence generation failed: {e}")

                # Store in history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": safe_answer_text,
                    "sources": redacted_docs,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "evidence_img": evidence_img,
                    "evidence_page": evidence_page,
                    "evidence_source": evidence_source,
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")

# --------- Display chat history ----------

if st.session_state.chat_history:
    st.markdown("---")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"**🙋 Question ({chat['timestamp']}):** {chat['question']}")
            st.markdown(f"**🤖 Answer:** {chat['answer']}")

            #Visual evidence (top source page with highlight)
           # if chat.get("evidence_img") is not None:
           #     src = chat.get("evidence_source") or "Unknown"
            #    page_no = (chat.get("evidence_page") or 0) + 1
             #   st.image(
              #      chat["evidence_img"],
               #     caption=f"📌 Visual evidence from {src} — Page {page_no}",
                #    use_container_width=True
                #)
          #  else:
           #     st.info(
            #        "Could not precisely locate the paragraph on the page, "
             #       "showing text sources only."
              #  )

            # Show sources
            with st.expander("📄 View Source Chunks"):
                for j, doc in enumerate(chat["sources"]):
                    source_name = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page_num')
                    page_info = f"(page {page + 1})" if page is not None else ""
                    st.text(f"Source {j+1} - {source_name} {page_info}:")
                    st.text(doc.page_content[:400] + "...")
                    st.markdown("---")

            st.markdown("---")
else:
    # Instructions when no documents uploaded
    st.info("👈 Please upload PDF documents from the sidebar to get started")

    st.markdown("### 🚀 How to Use")
    st.markdown("""
    1. Upload one or more PDF documents using the sidebar  
    2. Wait for secure in-memory processing to complete  
    3. Ask questions about the uploaded documents  
    4. Get answers with visual source proof (highlighted paragraphs)
    """)

    st.markdown("### ✨ Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔒 Secure**\n\nCompletely offline, local LLM via Ollama")
    with col2:
        st.markdown("**🧠 Zero-Trace**\n\nNo PDFs or indexes written to disk")
    with col3:
        st.markdown("**📌 Visual Evidence**\n\nShows exact source paragraph on original page")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>🇮🇳 Developed for DRDO | Internship Project 2026</div>",
    unsafe_allow_html=True
)