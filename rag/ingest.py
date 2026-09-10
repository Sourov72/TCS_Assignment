"""
Ingests every PDF in data/policies/ into FAISS vector store.


Pipeline per PDF:
  1. Load the PDF page-by-page (keeps page-number metadata for citations).
  2. Split each page's text into ~800-char chunks with 150-char overlap,
     using a recursive/paragraph-first splitter (robust to real-world PDFs
     that don't preserve heading structure once converted to plain text).
  3. Tag every chunk with `source_file` metadata (used for citations).
  4. Embed every chunk from every PDF into one combined FAISS index.

Run with:
    venv\\Scripts\\python.exe rag/ingest.py
"""

import hashlib
from pathlib import Path

from embeddings import get_embedding_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

POLICIES_DIR = Path(__file__).parent.parent / "data" / "policies"
INDEX_DIR = Path(__file__).parent / "vectorstore"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _load_pdf_pages(pdf_path: Path) -> list:
    """Load one LangChain Document per PDF page, each with page-number metadata."""
    return PyPDFLoader(str(pdf_path)).load()


def _chunk_pages(pages: list) -> list:
    """Split page documents into overlapping chunks, preserving page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)


def _tag_source_file(chunks: list, source_file: str) -> None:
    """Attach source_file metadata to each chunk, in place (used for citations)."""
    for chunk in chunks:
        chunk.metadata["source_file"] = source_file


def build_index() -> FAISS:
    """Load, chunk, and tag every PDF currently in data/policies/, then embed
    them all into one combined FAISS index."""
    all_chunks = []

    for pdf_path in sorted(POLICIES_DIR.glob("*.pdf")):
        pages = _load_pdf_pages(pdf_path)
        chunks = _chunk_pages(pages)
        _tag_source_file(chunks, pdf_path.name)
        all_chunks.extend(chunks)
        print(f"{pdf_path.name}: {len(pages)} page(s) -> {len(chunks)} chunk(s)")

    if not all_chunks:
        raise ValueError(f"No PDF files found in {POLICIES_DIR}")

    return FAISS.from_documents(all_chunks, get_embedding_model())


def ingest_all() -> None:
    """Build the combined index and persist it to disk.

    Note: this rebuilds the index from scratch, from whatever PDFs are
    currently in data/policies/ - it does not preserve documents added at
    runtime via add_pdf_to_index() unless they were saved into that same
    folder first (the Streamlit upload feature does save them there, so a
    re-ingest picks them up).
    """
    index = build_index()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(INDEX_DIR))
    print(f"Saved index ({index.index.ntotal} chunks) to {INDEX_DIR}")


def load_index() -> FAISS:
    """Load the persisted combined FAISS index from disk."""
    return FAISS.load_local(
        str(INDEX_DIR),
        get_embedding_model(),
        allow_dangerous_deserialization=True,  # safe: we only ever load our own index files
    )


def _file_hash(data: bytes) -> str:
    """SHA-256 hash of a file's raw bytes, used to detect duplicate uploads
    by content rather than filename (an upload gets a random-prefixed
    filename, so filename alone can never reveal a repeat)."""
    return hashlib.sha256(data).hexdigest()


def is_duplicate_pdf(file_bytes: bytes) -> bool:
    """Check whether file_bytes' content already matches an existing PDF in
    data/policies/ - used before ingesting an upload, so re-uploading the
    same document doesn't duplicate its chunks in the index."""
    new_hash = _file_hash(file_bytes)
    return any(
        _file_hash(existing.read_bytes()) == new_hash
        for existing in POLICIES_DIR.glob("*.pdf")
    )


def add_pdf_to_index(pdf_path: Path) -> int:
    """Chunk + tag + embed one new PDF and merge it into the combined index
    on disk (used by the Streamlit upload feature). Returns chunk count added."""
    pages = _load_pdf_pages(pdf_path)
    chunks = _chunk_pages(pages)
    _tag_source_file(chunks, pdf_path.name)

    index = load_index()
    index.add_documents(chunks)
    index.save_local(str(INDEX_DIR))

    return len(chunks)


if __name__ == "__main__":
    ingest_all()
