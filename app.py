
import logging
import os

from flask import Flask, render_template, request, jsonify
from chatbot import ChatbotError, ask_question, get_health_status

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    try:
        status = get_health_status()
        if not status["api_key_configured"] or status["document_count"] == 0:
            return jsonify({"status": "degraded", **status}), 503
        return jsonify({"status": "ok", **status}), 200
    except Exception:
        logger.exception("Health check failed")
        return jsonify({"status": "error"}), 500


@app.route("/ask", methods=["POST"])
def ask():

    try:
        data = request.get_json(silent=True) or {}

        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "error": "Please enter a question."
            }), 400

        answer = ask_question(question)

        return jsonify({
            "answer": answer
        })

    except ChatbotError as error:
        logger.exception("Chatbot request failed")
        return jsonify({
            "error": str(error)
        }), 502

    except Exception:
        logger.exception("Unexpected error in /ask")

        return jsonify({
            "error": "The server could not process your request."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
