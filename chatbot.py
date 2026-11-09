
import chromadb
import ollama
import time
import re
import json
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

CHAT_MODEL = "qwen3:4b"
EMBED_MODEL = "nomic-embed-text"

DB_PATH = "chroma_db"
COLLECTION_NAME = "legal_documents"

TOP_K = 8
MAX_TOKENS = 800
CONTEXT_SIZE = 4096
BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# 2. CONNECT TO CHROMADB
# ==========================================

client = chromadb.PersistentClient(path=str(BASE_DIR / DB_PATH))

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ==========================================
# 3. ASK QUESTION FUNCTION
# ==========================================

def _clean_response(response):
    """Return only the model's final answer when reasoning is included."""
    response = response.strip()

    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
            return parsed["answer"].strip()
    except json.JSONDecodeError:
        pass

    if "<think>" in response:
        response = response.split("<think>", 1)[-1]
        if "</think>" in response:
            response = response.split("</think>", 1)[-1]

    if "FINAL ANSWER:" in response:
        response = response.split("FINAL ANSWER:", 1)[-1]
    else:
        final_markers = re.findall(
            r"(?:Therefore|Thus|So),?\s+(?:the\s+)?answer\s+is:?\s*",
            response,
            flags=re.IGNORECASE,
        )
        if final_markers:
            response = re.split(
                r"(?:Therefore|Thus|So),?\s+(?:the\s+)?answer\s+is:?\s*",
                response,
                flags=re.IGNORECASE,
            )[-1]

    return response.strip()


def _prioritize_numbered_provision(question, documents, metadatas):
    match = re.search(r"\b(article|section)\s+(\d+[A-Za-z]?)\b", question, re.IGNORECASE)
    if not match:
        return documents, metadatas

    provision_number = match.group(2)
    exact_pattern = re.compile(
        rf"(?<!\d){re.escape(provision_number)}\s*\.",
        re.IGNORECASE,
    )
    stored = collection.get(include=["documents", "metadatas"])
    exact_documents = []
    exact_metadatas = []

    for document, metadata in zip(
        stored.get("documents") or [],
        stored.get("metadatas") or [],
    ):
        if document and exact_pattern.search(document):
            exact_documents.append(document)
            exact_metadatas.append(metadata or {})

    if not exact_documents:
        return documents, metadatas

    combined_documents = exact_documents + [
        document for document in documents if document not in exact_documents
    ]
    combined_metadatas = exact_metadatas + [
        metadata
        for document, metadata in zip(documents, metadatas)
        if document not in exact_documents
    ]
    return combined_documents[:TOP_K], combined_metadatas[:TOP_K]


def ask_question(question):

    try:

        question = question.strip()
        if not question:
            return "Please enter a legal question."

        total_start = time.time()

        # Check database
        if collection.count() == 0:
            return "Your legal knowledge base is empty. Build the database first."

        # --------------------------------------
        # 4. GENERATE QUESTION EMBEDDING
        # --------------------------------------

        print("\nGenerating question embedding...")

        embedding_start = time.time()

        response = ollama.embed(
            model=EMBED_MODEL,
            input=question,
            keep_alive="10m"
        )

        query_embedding = response["embeddings"][0]

        embedding_time = time.time() - embedding_start

        # --------------------------------------
        # 5. SEARCH RELEVANT DOCUMENTS
        # --------------------------------------

        print("Searching legal documents...")

        search_start = time.time()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(TOP_K, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        search_time = time.time() - search_start

        documents = results["documents"][0] or []
        metadatas = results["metadatas"][0] or []

        if not documents:
            return "No relevant information found in the legal documents."

        documents, metadatas = _prioritize_numbered_provision(
            question,
            documents,
            metadatas,
        )

        # --------------------------------------
        # 6. BUILD CONTEXT
        # --------------------------------------

        context = ""

        for i, doc in enumerate(documents):

            metadata = metadatas[i] or {}

            source = metadata.get(
                "source",
                "Unknown document"
            )

            page = metadata.get(
                "page",
                "Unknown page"
            )

            context += f"""
Source: {source}
Page: {page}

Content:
{doc}

------------------------------
"""

        # --------------------------------------
        # 7. GENERATE ANSWER USING QWEN3
        # --------------------------------------

        print("Generating legal answer...")

        generation_start = time.time()

  
       
        prompt = f"""
/no_think

You are an Indian Legal Information Assistant.

Answer the user's question using ONLY the provided legal context.

STRICT INSTRUCTIONS:

1. Return exactly one concise answer; do not show analysis or reasoning.
2. Do not repeat the question or discuss the retrieval process.
3. Do not list unrelated provisions.
4. Mention relevant Article or Section numbers.
5. Do not mention PDF names or page numbers.
6. Do not invent legal information.
7. If information is insufficient, say so directly.
8. Use simple, understandable language.
9. Keep the answer within 150 words.

Return valid JSON only, in exactly this format:
{{"answer": "Relevant Article or Section – Act Name\\n\\nDirect explanation in one or two short paragraphs."}}

LEGAL CONTEXT:
{context}

USER QUESTION:
{question}

FINAL ANSWER:
"""


        answer = ollama.generate(
            model=CHAT_MODEL,
            prompt=prompt,
            think=False,
            keep_alive="10m",
            format={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"}
                },
                "required": ["answer"]
            },
            options={
                "temperature": 0.2,
                "num_predict": MAX_TOKENS,
                "num_ctx": CONTEXT_SIZE
            }
        )

        generation_time = time.time() - generation_start


        # --------------------------------------
        # 8. CLEAN MODEL RESPONSE
        # --------------------------------------

        raw_response = _clean_response(answer.get("response", ""))

        if not raw_response:

            print("Warning: Model returned an empty response.")

            return (
                "The model returned an empty answer. "
                "Please check the Ollama model and debug output."
            )

        # --------------------------------------
        # 9. PERFORMANCE REPORT
        # --------------------------------------

        total_time = time.time() - total_start

        print("\n========== PERFORMANCE ==========")

        print(f"Embedding Time : {embedding_time:.2f} seconds")
        print(f"Search Time    : {search_time:.2f} seconds")
        print(f"Generation Time: {generation_time:.2f} seconds")
        print(f"Total Time     : {total_time:.2f} seconds")

        print("=================================")

        # --------------------------------------
        # 10. RETURN ANSWER
        # --------------------------------------

        return raw_response.strip()

    except Exception as e:

        print("\nERROR:", str(e))

        return f"An error occurred: {str(e)}"


# ==========================================
# 11. MAIN CHAT INTERFACE
# ==========================================

if __name__ == "__main__":

    print("\n======================================")
    print("          LEGAL RAG CHATBOT")
    print("======================================")

    print("Model:", CHAT_MODEL)
    print("Embedding:", EMBED_MODEL)

    print("\nType 'exit' to stop the chatbot.")

    while True:

        question = input("\nAsk your legal question: ").strip()

        if question.lower() == "exit":

            print("\nChatbot closed.")
            break

        if not question:

            print("Please enter a question.")
            continue

        print("\nProcessing your question...")

        answer = ask_question(question)

        print("\n========== CHATBOT ANSWER ==========")

        print(answer)

        print("====================================")
