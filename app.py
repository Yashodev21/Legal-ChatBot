import logging
import os
import time

from flask import Flask, render_template, request, jsonify
from chatbot import ChatbotError, ask_question, get_health_status

# ==========================================
# 1. CONFIGURATION
# ==========================================

app = Flask(__name__)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# Prevent excessively large request bodies.
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


# ==========================================
# 2. HOME PAGE
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ==========================================
# 3. HEALTH CHECK
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    try:
        status = get_health_status()

        if not status["api_key_configured"] or status["document_count"] == 0:
            return jsonify({
                "status": "degraded",
                **status
            }), 503

        return jsonify({
            "status": "ok",
            **status
        }), 200

    except Exception:
        logger.exception("Health check failed")

        return jsonify({
            "status": "error",
            "message": "Health check failed."
        }), 500


# ==========================================
# 4. ASK QUESTION API
# ==========================================

@app.route("/ask", methods=["POST"])
def ask():

    request_start = time.monotonic()

    logger.info("ASK ROUTE STARTED")

    try:
        # Validate JSON
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Invalid request. JSON body required."
            }), 400

        question = data.get("question", "")

        if not isinstance(question, str):
            return jsonify({
                "error": "Question must be text."
            }), 400

        question = question.strip()

        if not question:
            return jsonify({
                "error": "Please enter a legal question."
            }), 400

        if len(question) > 2000:
            return jsonify({
                "error": "Question is too long. Maximum 2000 characters."
            }), 400

        logger.info(
            "QUESTION RECEIVED: length=%d",
            len(question)
        )

        # Generate legal answer
        logger.info("CALLING CHATBOT SERVICE")

        chatbot_start = time.monotonic()

        answer = ask_question(question)

        chatbot_time = time.monotonic() - chatbot_start

        logger.info(
            "CHATBOT COMPLETED: %.2f seconds",
            chatbot_time
        )

        if not answer:
            return jsonify({
                "error": "The chatbot returned an empty answer."
            }), 500

        logger.info(
            "ASK ROUTE COMPLETED: total=%.2f seconds",
            time.monotonic() - request_start
        )

        return jsonify({
            "answer": answer
        }), 200

    except ChatbotError as error:

        logger.exception(
            "CHATBOT ERROR: %s",
            error
        )

        return jsonify({
            "error": str(error)
        }), 502

    except Exception:

        logger.exception(
            "UNEXPECTED ERROR IN ASK ROUTE"
        )

        return jsonify({
            "error": "The server could not process your request."
        }), 500


# ==========================================
# 5. ERROR HANDLERS
# ==========================================

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": "Request body is too large."
    }), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Requested endpoint was not found."
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "HTTP method not allowed."
    }), 405


# ==========================================
# 6. LOCAL DEVELOPMENT
# ==========================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", 5000)),
        debug=False
    )