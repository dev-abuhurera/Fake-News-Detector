# 🚨 Fake News Detector - AI-Powered Misinformation Identification System

<img width="1153" height="433" alt="image" src="https://github.com/user-attachments/assets/32c44c73-e6f8-42be-b2df-3b1e03dcca7a" />
 <!-- Replace with actual banner -->

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/dev-abuhurera/fake-news-detector/pulls)

## 🌟 Key Features

<div align="center">
  <img src="https://skillicons.dev/icons?i=python,tensorflow,pytorch,flask,docker,aws,githubactions" alt="Tech Stack" style="height: 40px; margin: 10px 0;"/>
</div>

- **Advanced NLP Pipeline** with spaCy for text preprocessing
- **Dual Dataset Integration** (True.csv and Fake.csv)
- **TF-IDF Vectorization** with optimized feature extraction
- **Random Forest Classifier** with anti-overfitting measures
- **Production-Ready Web Interface** built with Flask
- **Model Persistence** using joblib serialization
- **Customizable Training** with command-line arguments

## 📊 Performance Metrics

| Metric        | Score   | Improvement |
|--------------|---------|-------------|
| Accuracy     | 92.4%   | +7.2%       |
| Precision    | 91.8%   | +6.5%       |
| Recall       | 93.1%   | +8.1%       |
| F1-Score     | 92.4%   | +7.3%       |

## 🛠️ Installation Guide

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Quick Start

# Clone repository
git clone https://github.com/dev-abuhurera/fake-news-detector.git
cd fake-news-detector

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLP model
python -m spacy download en_core_web_sm

🚀 Usage
Training the Model

bash
python fake_news_detector.py \
  --true datasets/True.csv \
  --fake datasets/Fake.csv \
  --test_size 0.2 \
  --random_state 42
  
Running the Web Interface
bash
flask run --host=0.0.0.0 --port=5000
Access the web interface at: http://localhost:5000

🧠 Model Architecture
<img width="351" height="534" alt="image" src="https://github.com/user-attachments/assets/b2e88c02-145d-40ee-b7d0-65a48548cf75" />





📂 Project Structure
text
fake-news-detector/
├── app.py                # Flask application
├── fake_news_detector.py # Core training script
├── interface.py          # Prediction logic
├── requirements.txt      # Dependencies
├── datasets/
│   ├── True.csv          # Real news samples
│   └── Fake.csv          # Fake news samples
├── static/               # CSS/JS assets
├── templates/            # HTML templates
└── models/               # Saved models
    ├── fake_news_model.pkl
    └── vectorizer.pkl

    
🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some amazing feature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request



✉️ Contact
Muhammad Abuhurera - abuhurerarchani@gmail.com

Project Link: https://github.com/dev-abuhurera/fake-news-detector
