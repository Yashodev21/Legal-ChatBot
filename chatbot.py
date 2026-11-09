
import chromadb
import time
import re
import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class ChatbotError(RuntimeError):
    """Raised when the legal assistant cannot complete a request."""

# ==========================================
# 1. CONFIGURATION
# ==========================================

CHAT_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

DB_PATH = "chroma_db_gemini"
COLLECTION_NAME = "legal_documents_gemini"

TOP_K = 6
MAX_TOKENS = 350
API_TIMEOUT_MS = 12000
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# ==========================================
# 2. CONNECT TO CHROMADB
# ==========================================

client = chromadb.PersistentClient(path=str(BASE_DIR / DB_PATH))

try:
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception:
    collection = None


def _get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to the .env file or Render environment."
        )
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=API_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
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


def _is_ambiguous_section_query(question):
    return bool(
        re.search(r"\bsection\s+\d+[A-Za-z]?\b", question, re.IGNORECASE)
        and not re.search(
            r"\b(bns|bharatiya\s+nyaya\s+sanhita|act|law)\b",
            question,
            re.IGNORECASE,
        )
    )


def _prioritize_numbered_provision(question, documents, metadatas):
    match = re.search(r"\b(article|section)\s+(\d+[A-Za-z]?)\b", question, re.IGNORECASE)
    if not match:
        return documents, metadatas

    provision_number = match.group(2)
    exact_pattern = re.compile(
        rf"(?<!\d){re.escape(provision_number)}\s*\.",
        re.IGNORECASE,
    )
    if collection is None:
        return documents, metadatas

    stored = collection.get(
        where_document={"$contains": f"{provision_number}."},
        include=["documents", "metadatas"],
    )
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


def _lexical_retrieve(question):
    """Retrieve stored chunks without calling the embedding API."""
    if collection is None:
        return [], []

    stored = collection.get(include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9]+", question)
        if len(term) >= 3
    }
    phrase = question.strip().lower()
    scored = []

    for index, document in enumerate(documents):
        text = (document or "").lower()
        score = sum(text.count(term) for term in terms)
        if phrase and phrase in text:
            score += 10
        if "fundamental right" in phrase and "fundamental right" in text:
            score += 20
        if score:
            scored.append((score, index))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [index for _, index in scored[:TOP_K]]
    return (
        [documents[index] for index in selected],
        [metadatas[index] or {} for index in selected],
    )


def ask_question(question):

    try:

        question = question.strip()
        if not question:
            return "Please enter a legal question."

        if _is_ambiguous_section_query(question):
            return (
                "Which Act do you mean? Please specify the Bharatiya Nyaya Sanhita "
                "or the relevant Act name."
            )

        total_start = time.time()

        # Check database
        if collection is None or collection.count() == 0:
            return "The Gemini knowledge base is empty. Run build_database.py first."

        # --------------------------------------
        # 4. GENERATE QUESTION EMBEDDING
        # --------------------------------------

        print("\nGenerating question embedding...")

        embedding_start = time.time()

        gemini = _get_gemini_client()
        try:
            response = gemini.models.embed_content(
                model=EMBED_MODEL,
                contents=question,
            )

            if not response.embeddings or not response.embeddings[0].values:
                raise RuntimeError("Gemini returned an invalid question embedding.")

            query_embedding = response.embeddings[0].values
            embedding_time = time.time() - embedding_start
            used_fallback = False
        except Exception as error:
            embedding_time = time.time() - embedding_start
            logger.warning(
                "Gemini embedding unavailable; using lexical retrieval: %s",
                type(error).__name__,
            )
            query_embedding = None
            used_fallback = True

        # --------------------------------------
        # 5. SEARCH RELEVANT DOCUMENTS
        # --------------------------------------

        print("Searching legal documents...")

        search_start = time.time()

        if used_fallback:
            documents, metadatas = _lexical_retrieve(question)
        else:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(TOP_K, collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            documents = results["documents"][0] or []
            metadatas = results["metadatas"][0] or []

        search_time = time.time() - search_start

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

            context += f"Document chunk {i + 1}:\n{doc[:900]}\n\n"

        # --------------------------------------
        # 7. GENERATE ANSWER USING GEMINI
        # --------------------------------------

        print("Generating legal answer...")

        generation_start = time.time()

  
       
        prompt = f"""
You are an Indian legal information assistant. Use only the context below.
Return JSON only: {{"answer":"Article/Section - Act Name\\n\\nConcise answer."}}
Do not include reasoning, source names, page numbers, or retrieval details.
If the context is insufficient, say so. Never invent provisions.

CONTEXT:
{context}

QUESTION:
{question}
"""


        answer = gemini.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=MAX_TOKENS,
                response_mime_type="application/json",
            ),
        )

        generation_time = time.time() - generation_start


        # --------------------------------------
        # 8. CLEAN MODEL RESPONSE
        # --------------------------------------

        raw_response = _clean_response(answer.text or "")

        if not raw_response:

            print("Warning: Model returned an empty response.")

            raise ChatbotError(
                "Gemini returned an empty answer. Please try again."
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

    except ChatbotError:
        raise
    except Exception as error:
        logger.exception("Legal assistant request failed")
        raise ChatbotError(
            "The legal assistant is temporarily unavailable. Please try again."
        ) from error


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
