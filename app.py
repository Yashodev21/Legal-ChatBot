
from flask import Flask, render_template, request, jsonify
from chatbot import ask_question

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


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

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
