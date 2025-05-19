from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import string
import re

from transformers import MarianMTModel, MarianTokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

# Load your fine-tuned model
model_path = "./masbatenyo_bidirectional2_model"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/translator")
def translator():
    return render_template("translator.html")

@app.route("/general-conversations")
def general_conversations():
    return render_template("general conversations.html")

@app.route("/favorites")
def favorites():
    return render_template("favorites.html")

@app.route("/privacy-policy")
def privacy_policy():
    return render_template("privacy policy.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    input_text = data.get("text", "")
    direction = data.get("direction", "en-to-masbatenyo")

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400
    
    # Detect punctuation in input
    has_punctuation = bool(re.search(r'[.?!"]', input_text))

    # Translate accordingly
    if direction == "en-to-masbatenyo":
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs)
        output_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    else:
        # Masbatenyo → English
        # Just swap tokenizer input/output — assumes tokenizer can handle it
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs)
        output_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # Remove punctuation from output only if input has no punctuation
    if not has_punctuation:
        output_text = ''.join(ch for ch in output_text if ch not in ['"', '?', '.', '!'])

    # Remove unwanted punctuation characters: ", ?, ., !
    unwanted_chars = ['"']
    output_text = ''.join(ch for ch in output_text if ch not in unwanted_chars)

    return jsonify({"translation": output_text})

if __name__ == "__main__":
    app.run(debug=True)
