from flask import Flask, render_template, request, jsonify
import joblib
import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "26f1f696e7b8735398380ef1ed7c04a0f3462d850410c2c4"

@app.route("/")
def index():
    return render_template("index.html")  # Render the main page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        news_text = request.form["news"]
        if not news_text.strip():
            return jsonify({"error": "Please enter some text."})

        cleaned_text = preprocess_text(news_text)
        transformed_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(transformed_text)[0]
        prediction = "Real News" if prediction == 1 else "Fake News"

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)