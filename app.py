<<<<<<< HEAD
import os
import argparse
import joblib
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure spaCy model is installed
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        print("Downloading 'en_core_web_sm' model...")
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])

nlp = load_spacy_model()

# Define preprocessing function using spaCy
def preprocess_text_spacy(text):
    text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def main(true_path, fake_path, model_path, vectorizer_path):
    # Load datasets
    true = pd.read_csv(true_path)
    fake = pd.read_csv(fake_path)
    
    # Assign labels
    true['label'] = 1  # 1 for real news
    fake['label'] = 0  # 0 for fake news
    
    # Combine into one DataFrame
    data = pd.concat([true, fake], ignore_index=True)
    
    # Ensure 'text' column exists
    if 'text' not in data.columns:
        if 'title' in data.columns:
            data['text'] = data['title'] + " " + data.get('text', '')
        else:
            raise ValueError("No 'text' or similar column found. Check dataset columns.")
    
    # Apply preprocessing using spaCy with optimized memory handling
    print("Preprocessing text...")
    texts = data['text'].tolist()
    processed_texts = [preprocess_text_spacy(text) for text in nlp.pipe(texts, batch_size=1000)]
    data['cleaned_text'] = processed_texts
    
    # Vectorize and train
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_text'])
    y = data['label']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2,       # Prevent overfitting
        max_features='sqrt'       # Consider sqrt(n_features) at each split
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    rf_y_pred = rf_model.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))
    
    # Save model and vectorizer
    joblib.dump(rf_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Model saved at: {model_path}")
    print(f"Vectorizer saved at: {vectorizer_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News Detection Script")
    parser.add_argument("--true", type=str, required=True, help="Path to True news dataset")
    parser.add_argument("--fake", type=str, required=True, help="Path to Fake news dataset")
    parser.add_argument("--model", type=str, default="fake_news_model.pkl", help="Path to save trained model")
    parser.add_argument("--vectorizer", type=str, default="vectorizer.pkl", help="Path to save vectorizer")
    args = parser.parse_args()
    
    main(args.true, args.fake, args.model, args.vectorizer)
=======
import os
import joblib
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load spaCy model........
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Disable unnecessary components for faster processing

# Define preprocessing function using spaCy.....
def preprocess_text_spacy(text):
    text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Get absolute paths dynamically....

base_dir = os.path.dirname(os.path.abspath(__file__))
true_path = os.path.join(base_dir, "True.csv")
fake_path = os.path.join(base_dir, "Fake.csv")
model_path = os.path.join(base_dir, "fake_news_model.pkl")
vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")

for path in [true_path, fake_path, model_path, vectorizer_path]:
    if os.path.exists(path):
        print(f"File exists: {path}")
    else:
        print(f"File does not exist: {path}")

# Load datasets
true = pd.read_csv(true_path)
fake = pd.read_csv(fake_path)

print("true shape", true.shape)
print("fake shape", fake.shape)
print(true.head())
print(fake.head())

# Assign labels
true['label'] = 1  # 1 for real news
fake['label'] = 0  # 0 for fake news

# Combine into one DataFrame
data = pd.concat([true, fake], ignore_index=True)

# Ensure 'text' column exists
if 'text' not in data.columns:
    if 'title' in data.columns and 'text' not in data.columns:
        data['text'] = data['title'] + " " + data.get('text', '')
    else:
        raise ValueError("No 'text' or similar column found. Check dataset columns.")

# Apply preprocessing using spaCy with multi-threading
print("Preprocessing text...")
texts = data['text'].tolist()
if __name__ == "__main__":
    processed_texts = [preprocess_text_spacy(text) for text in nlp.pipe(texts)] # Use 2 threads
data['cleaned_text'] = processed_texts

# Vectorize and train
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    min_samples_leaf=2,       # Prevent overfitting
    max_features='sqrt'       # Consider sqrt(n_features) at each split
)
rf_model.fit(X_train, y_train)

# Evaluate
rf_y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))

# Test a sample
test_article = "This is a fake news article about something untrue."
cleaned_test = preprocess_text_spacy(test_article)
vectorized_test = vectorizer.transform([cleaned_test])
print("Random Forest Prediction:", "Fake" if rf_model.predict(vectorized_test)[0] == 0 else "Real")

# Save model and vectorizer
joblib.dump(rf_model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Model saved at: {model_path}")
print(f"Vectorizer saved at: {vectorizer_path}")


# #///////////////////////////////////////////////////////////////////////////////////////////////////////////#

>>>>>>> c3ee8fa (Added new files and made changes to Fake-news-detector)
