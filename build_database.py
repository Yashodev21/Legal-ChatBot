
import os
import re
import uuid
import chromadb
import ollama

DATA_FOLDER = "extracted_text"
DB_FOLDER = "chroma_db"

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize ChromaDB
client = chromadb.PersistentClient(path=DB_FOLDER)

collection = client.get_or_create_collection(
    name="legal_documents"
)

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


def process_documents():
    for filename in os.listdir(DATA_FOLDER):

        if not filename.endswith(".txt"):
            continue

        path = os.path.join(DATA_FOLDER, filename)

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        # Split text while retaining page references
        pages = re.split(r"--- Page (\d+) ---", text)

        for i in range(1, len(pages), 2):
            page_number = int(pages[i])
            page_text = pages[i + 1]

            chunks = split_into_chunks(page_text)

            for chunk in chunks:

                response = ollama.embed(
                    model=EMBED_MODEL,
                    input=chunk
                )

                embedding = response["embeddings"][0]

                collection.add(
                    ids=[str(uuid.uuid4())],
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": filename,
                        "page": page_number
                    }]
                )

        print(f"Processed: {filename}")

    print("Knowledge base created successfully!")
    print("Total chunks:", collection.count())


if __name__ == "__main__":
    process_documents()
