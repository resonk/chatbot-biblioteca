from flask import Flask, render_template, request, jsonify
from chatbot import responder

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        pregunta = data.get("mensaje", "").strip()

        if not pregunta:
            return jsonify({"respuesta": "Debes escribir una pregunta."}), 400

        respuesta = responder(pregunta)
        return jsonify({"respuesta": respuesta})

    except Exception as e:
        return jsonify({"respuesta": f"Error interno: {str(e)}"}), 500