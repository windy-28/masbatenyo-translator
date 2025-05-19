from flask import Flask, render_template, request, jsonify
import os
import string
import re
import gdown
import zipfile
import requests

from transformers import MarianMTModel, MarianTokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

# Google Drive info for zipped model
ZIP_PATH = "masbatenyo_bidirectional2_model.zip"
EXTRACT_DIR = "masbatenyo_bidirectional2_model"
FILE_ID = "18vrn0FiH5WMn4K_hUAFmKxBjcA-A0RwT"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

def download_and_extract_model():
    if not os.path.exists(EXTRACT_DIR):
        print("Downloading zipped model from Google Drive...")
        gdown.download(URL, ZIP_PATH, quiet=False)
        print("Extracting model...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Model extracted.")
    else:
        print("Model already downloaded and extracted.")

# After extracting the zip
print("Listing extracted model directory:")
for root, dirs, files in os.walk("masbatenyo_bidirectional2_model"):
    print("DIR:", root)
    for f in files:
        print("   FILE:", f)


# Download and extract model before loading
download_and_extract_model()

# Load tokenizer and model from extracted folder
tokenizer = MarianTokenizer.from_pretrained(EXTRACT_DIR)
model = MarianMTModel.from_pretrained(EXTRACT_DIR)

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
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
