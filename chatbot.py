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

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class ChatbotError(RuntimeError):
    pass


class ChatbotConfigurationError(ChatbotError):
    pass


class ChatbotServiceError(ChatbotError):
    pass


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = "gemini-embedding-001"
DB_PATH = BASE_DIR / "chroma_db_gemini"
COLLECTION_NAME = "legal_documents_gemini"
TOP_K = 4
MAX_TOKENS = 300
API_TIMEOUT_MS = 20000


# Connect without calling collection.count().
logger.info("Connecting to ChromaDB...")
try:
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    logger.info("ChromaDB collection connected: %s", COLLECTION_NAME)
except Exception:
    logger.exception("ChromaDB initialization failed")
    collection = None


def get_health_status():
    # Do not call collection.count() during health checks.
    return {
        "api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBED_MODEL,
        "database_collection": COLLECTION_NAME,
        "database_connected": collection is not None,
        "document_count": None,
    }


def _get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ChatbotConfigurationError("GEMINI_API_KEY is not configured.")

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=API_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _clean_response(response):
    response = (response or "").strip()
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

    return response.strip()


def _is_ambiguous_section_query(question):
    has_section = re.search(
        r"\\bsection\\s+\\d+[A-Za-z]?\\b", question, re.IGNORECASE
    )
    has_act = re.search(
        r"\\b(bns|bharatiya\\s+nyaya\\s+sanhita|act|law)\\b",
        question,
        re.IGNORECASE,
    )
    return bool(has_section and not has_act)


def _prioritize_numbered_provision(
    question, query_embedding, documents, metadatas
):
    match = re.search(
        r"\\b(article|section)\\s+(\\d+[A-Za-z]?)\\b",
        question,
        re.IGNORECASE,
    )
    if not match or collection is None:
        return documents, metadatas

    provision_number = match.group(2)
    exact_pattern = re.compile(
        rf"(?<!\\d){re.escape(provision_number)}\\s*\\.",
        re.IGNORECASE,
    )

    logger.info("EXACT PROVISION SEARCH STARTED: %s", provision_number)
    try:
        # No collection.count() call. Query the requested top K directly.
        stored = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            where_document={"$contains": f"{provision_number}."},
            include=["documents", "metadatas"],
        )

        exact_documents = []
        exact_metadatas = []
        result_documents = (stored.get("documents") or [[]])[0]
        result_metadatas = (stored.get("metadatas") or [[]])[0]

        for document, metadata in zip(result_documents, result_metadatas):
            if document and exact_pattern.search(document):
                exact_documents.append(document)
                exact_metadatas.append(metadata or {})

        logger.info(
            "EXACT PROVISION SEARCH COMPLETED: %d matches",
            len(exact_documents),
        )
        if not exact_documents:
            return documents, metadatas

        combined_documents = exact_documents + [
            doc for doc in documents if doc not in exact_documents
        ]
        combined_metadatas = exact_metadatas + [
            metadata
            for doc, metadata in zip(documents, metadatas)
            if doc not in exact_documents
        ]
        return combined_documents[:TOP_K], combined_metadatas[:TOP_K]

    except Exception:
        logger.exception("Exact provision retrieval failed")
        return documents, metadatas


def ask_question(question):
    total_start = time.monotonic()

    try:
        question = (question or "").strip()
        if not question:
            return "Please enter a legal question."

        if _is_ambiguous_section_query(question):
            return (
                "Which Act do you mean? Please specify "
                "the Bharatiya Nyaya Sanhita or the relevant Act name."
            )

        logger.info("DATABASE CONNECTION CHECK STARTED")
        if collection is None:
            raise ChatbotConfigurationError(
                "Legal knowledge base is unavailable."
            )
        # Deliberately skip collection.count(), which was hanging on Render.
        logger.info("DATABASE CONNECTION CHECK COMPLETED")

        logger.info("GEMINI CLIENT INITIALIZATION STARTED")
        gemini = _get_gemini_client()
        logger.info("GEMINI CLIENT INITIALIZATION COMPLETED")

        embedding_start = time.monotonic()
        logger.info("GEMINI EMBEDDING STARTED")
        response = gemini.models.embed_content(
            model=EMBED_MODEL,
            contents=question,
        )
        logger.info("GEMINI EMBEDDING RESPONSE RECEIVED")

        if not response.embeddings or not response.embeddings[0].values:
            raise ChatbotServiceError("Gemini returned an invalid embedding.")

        query_embedding = response.embeddings[0].values
        embedding_time = time.monotonic() - embedding_start
        logger.info("GEMINI EMBEDDING COMPLETED: %.2f seconds", embedding_time)

        search_start = time.monotonic()
        logger.info("CHROMADB SEARCH STARTED")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        search_time = time.monotonic() - search_start

        logger.info(
            "CHROMADB SEARCH COMPLETED: %d results, %.2f seconds",
            len(documents),
            search_time,
        )
        if not documents:
            return "No relevant information found in the legal documents."

        documents, metadatas = _prioritize_numbered_provision(
            question, query_embedding, documents, metadatas
        )

        logger.info("CONTEXT BUILDING STARTED")
        context_parts = [
            f"Document chunk {index + 1}:\\n{document[:700]}"
            for index, document in enumerate(documents)
        ]
        context = "\\n\\n".join(context_parts)
        logger.info(
            "CONTEXT BUILDING COMPLETED: %d characters", len(context)
        )

        prompt = f"""
You are an Indian legal information assistant.

Use only the legal context provided below.

RULES:
1. Give a concise and accurate answer.
2. Mention the correct Section or Article number.
3. Mention the applicable Act name.
4. Do not invent legal provisions.
5. Do not reveal internal reasoning.
6. Do not mention PDF names or page numbers.
7. If the context is insufficient, clearly say so.
8. Return valid JSON only.

Required JSON format:
{{"answer": "Section/Article - Act Name\\n\\nLegal answer."}}

LEGAL CONTEXT:
{context}

USER QUESTION:
{question}
"""

        logger.info("GEMINI GENERATION STARTED")
        generation_start = time.monotonic()
        answer = gemini.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=MAX_TOKENS,
                response_mime_type="application/json",
            ),
        )
        generation_time = time.monotonic() - generation_start
        logger.info(
            "GEMINI GENERATION COMPLETED: %.2f seconds", generation_time
        )

        logger.info("RESPONSE CLEANING STARTED")
        raw_response = _clean_response(answer.text or "")
        if not raw_response:
            raise ChatbotServiceError("Gemini returned an empty answer.")
        logger.info("RESPONSE CLEANING COMPLETED")

        total_time = time.monotonic() - total_start
        logger.info(
            "PERFORMANCE: embedding=%.2fs retrieval=%.2fs "
            "generation=%.2fs total=%.2fs",
            embedding_time,
            search_time,
            generation_time,
            total_time,
        )
        return raw_response

    except ChatbotError:
        raise
    except Exception as error:
        logger.exception("LEGAL CHATBOT REQUEST FAILED")
        error_text = str(error).lower()

        if (
            "429" in error_text
            or "quota" in error_text
            or "resource_exhausted" in error_text
        ):
            message = "Gemini API quota exceeded. Please try again later."
        elif "timeout" in error_text or "timed out" in error_text:
            message = "Gemini request timed out. Please try again."
        elif "404" in error_text:
            message = "Configured Gemini model is unavailable."
        else:
            message = (
                "The legal chatbot could not process your question. "
                "Check server logs."
            )
        raise ChatbotServiceError(message) from error


if __name__ == "__main__":
    print("\\nLEGAL RAG CHATBOT")
    print("Model:", CHAT_MODEL)
    print("Embedding:", EMBED_MODEL)
    print("\\nType 'exit' to stop.")

    while True:
        question = input("\\nAsk your legal question: ").strip()
        if question.lower() == "exit":
            break
        try:
            print("\\nCHATBOT ANSWER:\\n")
            print(ask_question(question))
        except ChatbotError as error:
            print("\\nERROR:", error)
