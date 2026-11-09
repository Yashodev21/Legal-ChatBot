import os
import re
import hashlib
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "extracted_text"
DB_FOLDER = BASE_DIR / "chroma_db_gemini"
COLLECTION_NAME = "legal_documents_gemini"

EMBED_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 100

load_dotenv(BASE_DIR / ".env")


def split_into_chunks(text):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def _get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to the .env file before building."
        )
    return genai.Client(api_key=api_key)


def process_documents(rebuild=False):
    chroma = chromadb.PersistentClient(path=str(DB_FOLDER))
    collection = chroma.get_or_create_collection(name=COLLECTION_NAME)
    document_paths = sorted(DATA_FOLDER.glob("*.txt"))

    if not document_paths:
        raise FileNotFoundError(f"No extracted documents found in: {DATA_FOLDER}")

    if not rebuild and collection.count() > 0:
        print(
            "Gemini database is incomplete; adding only missing document chunks."
        )

    if rebuild:
        chroma.delete_collection(name=COLLECTION_NAME)
        collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

    existing = collection.get(include=["documents", "metadatas", "embeddings"])
    existing_keys = {
        (
            metadata.get("source"),
            metadata.get("page"),
            metadata.get("chunk"),
        )
        for metadata in (existing.get("metadatas") or [])
        if metadata
    }

    if not DATA_FOLDER.exists():
        raise FileNotFoundError(f"Document folder not found: {DATA_FOLDER}")

    existing_embeddings = existing.get("embeddings")
    existing_dimensions = {
        len(embedding)
        for embedding in (
            existing_embeddings if existing_embeddings is not None else []
        )
        if embedding is not None
    }
    if len(existing_dimensions) > 1:
        raise RuntimeError(
            f"Existing Gemini vectors have inconsistent dimensions: {existing_dimensions}."
        )

    embedding_dimension = next(iter(existing_dimensions), None)

    pending_documents = []
    for path in document_paths:
        text = path.read_text(encoding="utf-8")
        pages = re.split(r"--- Page (\d+) ---", text)
        pending_chunks = []

        for i in range(1, len(pages), 2):
            page_number = int(pages[i])
            page_chunks = split_into_chunks(pages[i + 1])

            for chunk_number, chunk in enumerate(page_chunks):
                key = (path.name, page_number, chunk_number)
                if key not in existing_keys:
                    pending_chunks.append((page_number, chunk_number, chunk))

        if not pending_chunks:
            print(f"Skipped existing: {path.name}")
            continue

        pending_documents.append((path, pending_chunks))

    if not pending_documents:
        print(
            f"Gemini database already contains {collection.count()} chunks; "
            "skipping rebuild."
        )
        print("Use `python build_database.py --rebuild` after changing source documents.")
        return

    gemini = _get_gemini_client()

    for path, pending_chunks in pending_documents:
        for start in range(0, len(pending_chunks), EMBED_BATCH_SIZE):
            batch = pending_chunks[start:start + EMBED_BATCH_SIZE]
            response = gemini.models.embed_content(
                model=EMBED_MODEL,
                contents=[item[2] for item in batch],
            )
            embeddings = response.embeddings or []

            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Gemini returned {len(embeddings)} embeddings for {len(batch)} chunks."
                )

            dimensions = {
                len(embedding.values)
                for embedding in embeddings
                if embedding.values
            }
            if len(dimensions) != 1:
                raise RuntimeError(
                    f"Gemini returned inconsistent embedding dimensions: {dimensions}."
                )
            batch_dimension = dimensions.pop()
            if (
                embedding_dimension is not None
                and batch_dimension != embedding_dimension
            ):
                raise RuntimeError(
                    "Gemini embedding dimension changed during database creation: "
                    f"{embedding_dimension} -> {batch_dimension}."
                )
            embedding_dimension = batch_dimension

            collection.add(
                ids=[
                    hashlib.sha256(
                        f"{path.name}:{item[0]}:{item[1]}".encode("utf-8")
                    ).hexdigest()
                    for item in batch
                ],
                documents=[item[2] for item in batch],
                embeddings=[embedding.values for embedding in embeddings],
                metadatas=[
                    {
                        "source": path.name,
                        "page": item[0],
                        "chunk": item[1],
                    }
                    for item in batch
                ],
            )
            existing_keys.update(
                (path.name, item[0], item[1])
                for item in batch
            )

        print(f"Processed: {path.name} ({len(pending_chunks)} chunks)")

    print("Gemini knowledge base created successfully!")
    print("Database:", DB_FOLDER)
    print("Collection:", COLLECTION_NAME)
    print("Total chunks:", collection.count())


if __name__ == "__main__":
    process_documents(rebuild="--rebuild" in sys.argv)
