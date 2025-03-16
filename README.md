Fake News Detector

This project is an AI-based model that can detect whether a news article is fake or real using machine learning. The project includes a Flask-based web interface for user interaction.

What You Need
✅ Python Installed (Recommended version 3.8 or later)

✅ Required Libraries (Install them later):

pandas

numpy

sklearn

flask

spacy

joblib

✅ Datasets Required:

True.csv (Real news dataset)

Fake.csv (Fake news dataset)

✅ Additional Files:

app.py (Flask application for the web interface)

interface.py (Interface logic)

fake_news_model.pkl (Pre-trained model)

vectorizer.pkl (TF-IDF vectorizer)

templates/ (HTML templates for the web interface)

static/ (CSS and other static files)

Step-by-Step Setup
1. Get the Project
Download or clone the repository from GitHub.

Open the folder where the project is saved.

2. Set Up the Environment
Create a separate Python environment to keep the project organized.

Activate the environment before running the project.

3. Install Required Libraries
Run the following command to install dependencies:

bash
Copy
pip install -r requirements.txt
(This will install all required libraries automatically.)

4. Download the Required Model
The project uses spaCy for text processing. If you haven’t installed the necessary language model, run:

bash
Copy
python -m spacy download en_core_web_sm
(This will download the required NLP model.)

5. Run the Application
Start training the model by running:

bash
Copy
python fake_news_detector.py --true path/to/True.csv --fake path/to/Fake.csv
(Replace path/to/True.csv and path/to/Fake.csv with actual file paths.)

6. Open the Web App
Use a web browser to access the interface and enter news text.

The web interface is served by Flask and can be accessed at http://127.0.0.1:5000/.

7. Use the Model
Enter or upload a news article.

Click the Predict button.

The model will classify the news as Fake or Real.

8. (Optional) Train the Model Again
You can retrain the model using different datasets if needed.

The trained model and vectorizer are saved in:

fake_news_model.pkl

vectorizer.pkl

If you want to retrain the model, simply delete these files and rerun the training command.

9. Close the Environment After Use
If you used a virtual environment, deactivate it when you’re done:

bash
Copy
deactivate
Changes and Improvements in This Version
🔹 Added NLP Preprocessing Using spaCy

The text is cleaned, tokenized, and lemmatized before training.

Stop words are removed for better accuracy.

🔹 Optimized Data Handling

The script ensures that the "text" column exists in the dataset.

If missing, it combines the "title" column as a fallback.

🔹 Improved Training Process

Uses TF-IDF Vectorization to convert text into numerical form.

Uses RandomForestClassifier with better hyperparameters for accuracy.

Prevents overfitting using min_samples_leaf=2 and max_features='sqrt'.

🔹 Ensured Model and Vectorizer Are Saved Properly

Saves both in pickle format (.pkl) for easy reusability.

🔹 User Can Choose Testing Data

Instead of hardcoding, users provide their own dataset paths while running the script.

🔹 Added Flask Web Interface

The project now includes a web interface built with Flask, HTML, and CSS.

Users can interact with the model through a user-friendly web page.
