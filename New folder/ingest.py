"""
ingest.py
---------
Handles the INDEXING PHASE of the RAG pipeline:
  1. Load PDF pages
  2. Split into overlapping chunks
  3. Embed chunks using a local HuggingFace model
  4. Store vectors in ChromaDB (persisted to disk)

Run standalone to pre-index a file:
    python ingest.py path/to/your.pdf
"""

import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Constants ────────────────────────────────────────────────────────────────
CHROMA_DIR       = "./chroma_db"          # where ChromaDB persists data on disk
COLLECTION_NAME  = "pdf_documents"        # logical name for this vector collection
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"    # lightweight, runs locally, no API key needed
CHUNK_SIZE       = 1000                   # characters per chunk
CHUNK_OVERLAP    = 200                    # overlap keeps context across chunk boundaries


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the local embedding model.
    First call downloads ~80 MB to ~/.cache/huggingface.
    Subsequent calls load from cache instantly.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},      # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_pdf(pdf_path: str, progress_callback=None) -> int:
    """
    Index a single PDF file into ChromaDB.

    Args:
        pdf_path:          Path to the PDF file.
        progress_callback: Optional callable(step: str, detail: str) for UI updates.

    Returns:
        Number of chunks indexed.

    Why each step exists
    --------------------
    LOAD   – PyPDFLoader reads each page as a separate Document object,
             preserving page numbers in metadata (used later for citations).

    CHUNK  – LLMs have context-window limits and retrieval works better on
             small, focused text. RecursiveCharacterTextSplitter tries to
             split on paragraph breaks first, then line breaks, then words,
             to keep chunks semantically coherent.

    EMBED  – We convert each chunk to a dense vector (list of floats).
             "all-MiniLM-L6-v2" produces 384-dimensional vectors.
             Two semantically similar sentences will have vectors that are
             close together in that 384-dimensional space.

    STORE  – ChromaDB saves both the raw text AND the vector for each chunk.
             It also stores metadata (source file, page number) so we can
             show citations later.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("load", f"Reading {os.path.basename(pdf_path)}")

    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()

    # Attach the source filename to every page's metadata
    for page in pages:
        page.metadata["source_file"] = os.path.basename(pdf_path)

    # ── Step 2: Chunk ────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("chunk", f"Splitting {len(pages)} pages into chunks")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on these separators in order; fall back to the next one
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # ── Step 3 + 4: Embed & Store ─────────────────────────────────────────────
    if progress_callback:
        progress_callback("embed", f"Embedding {len(chunks)} chunks (may take a moment…)")

    embeddings   = get_embeddings()
    vectorstore  = Chroma.from_documents(
        documents        = chunks,
        embedding        = embeddings,
        collection_name  = COLLECTION_NAME,
        persist_directory= CHROMA_DIR,
    )

    if progress_callback:
        progress_callback("done", f"Indexed {len(chunks)} chunks from {os.path.basename(pdf_path)}")

    return len(chunks)


def get_vectorstore() -> Chroma:
    """
    Connect to an existing ChromaDB collection (must have run ingest first).
    Used by retriever.py and app.py to load the already-indexed data.
    """
    embeddings = get_embeddings()
    return Chroma(
        collection_name  = COLLECTION_NAME,
        embedding_function = embeddings,
        persist_directory= CHROMA_DIR,
    )


def collection_exists() -> bool:
    """Return True if the ChromaDB collection has been populated."""
    if not os.path.exists(CHROMA_DIR):
        return False
    try:
        vs    = get_vectorstore()
        count = vs._collection.count()
        return count > 0
    except Exception:
        return False


def clear_collection():
    """Delete all vectors from the collection (used when re-indexing a new PDF)."""
    import shutil
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)


# ── CLI usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    def cli_callback(step, detail):
        icons = {"load": "📄", "chunk": "✂️", "embed": "🔢", "done": "✅"}
        print(f"{icons.get(step, '•')} {detail}")

    n = ingest_pdf(pdf_path, progress_callback=cli_callback)
    print(f"\nDone! {n} chunks are now searchable.")
