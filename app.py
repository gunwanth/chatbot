from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
import uuid
from datetime import datetime

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Store active chats in memory (maps session_id to chat object)
active_chats = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/new-chat", methods=["POST"])
def new_chat():
    session_id = str(uuid.uuid4())
    active_chats[session_id] = model.start_chat(history=[])
    return jsonify({"session_id": session_id})

@app.route("/chat", methods=["POST"])
def chat_response():
    user_input = request.json.get("message")
    session_id = request.json.get("session_id")
    
    if not user_input or not session_id:
        return jsonify({"error": "No message or session provided"}), 400
    
    if session_id not in active_chats:
        active_chats[session_id] = model.start_chat(history=[])

    try:
        chat = active_chats[session_id]
        response_text = chat.send_message(user_input).text

        if "code" in response_text.lower():
            response_text = f"Here is the code block:\n```\n{response_text}\n```"
        
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
            summary = df.describe().to_dict()  
            return jsonify({"response": "File uploaded successfully", "data_summary": summary})

        return jsonify({"response": "File uploaded successfully but not analyzed (unsupported format)."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
