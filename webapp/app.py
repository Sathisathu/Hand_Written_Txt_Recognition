import os
from flask import Flask, request, render_template, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
from model_server import model_server

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "files" not in request.files:
            session['results'] = []
        else:
            uploaded_files = request.files.getlist("files")
            results = []
            for file in uploaded_files:
                if file.filename == "":
                    continue
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_DIR, filename)
                file.save(filepath)
                try:
                    text = model_server.predict(filepath)
                    text = text.replace("|", "")  # Remove pipe symbols
                    results.append({"filename": filename, "text": text})
                except Exception as e:
                    results.append({"filename": filename, "text": f"Error: {str(e)}"})
            session['results'] = results
        return redirect(url_for("index"))

    # GET request
    results = session.get('results', None)
    session.pop('results', None)  # Clear after reading
    return render_template("index.html", results=results)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or request.files["image"].filename == "":
        return {"error": "no image uploaded"}, 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    try:
        text = model_server.predict(path)
    except Exception as e:
        return {"error": str(e)}, 500

    return {"text": text}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
