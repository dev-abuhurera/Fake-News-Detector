# 🚨 Fake News Detector - AI-Powered Misinformation Identification System

![Project Banner](https://github.com/user-attachments/assets/32c44c73-e6f8-42be-b2df-3b1e03dcca7a)

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/240304586-218f11fa-42f4-4af1-9e0a-a2a8e8a7f4e9.gif" width="100%" height="2px"/>
</div>

<div align="center">
  <a href="https://github.com/dev-abuhurera/fake-news-detector/stargazers">
    <img src="https://img.shields.io/github/stars/dev-abuhurera/fake-news-detector?style=for-the-badge&color=7d40ff" alt="GitHub Stars">
  </a>
  <a href="https://github.com/dev-abuhurera/fake-news-detector/issues">
    <img src="https://img.shields.io/github/issues/dev-abuhurera/fake-news-detector?style=for-the-badge&color=7d40ff" alt="GitHub Issues">
  </a>
  <a href="https://github.com/dev-abuhurera/fake-news-detector/network/members">
    <img src="https://img.shields.io/github/forks/dev-abuhurera/fake-news-detector?style=for-the-badge&color=7d40ff" alt="GitHub Forks">
  </a>
</div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/240304586-218f11fa-42f4-4af1-9e0a-a2a8e8a7f4e9.gif" width="100%" height="2px"/>
</div>

## 🌟 Key Features

<div align="center" style="margin: 20px 0;">
  <img src="https://skillicons.dev/icons?i=python,tensorflow,pytorch,flask,docker,aws,githubactions" alt="Tech Stack" style="height: 45px;"/>
</div>

<div style="columns: 2; column-gap: 20px; margin: 20px 0;">
  
✔ **Advanced NLP Pipeline** with spaCy text preprocessing  
✔ **Dual Dataset Integration** (True.csv and Fake.csv)  
✔ **TF-IDF Vectorization** with optimized feature extraction  
✔ **Random Forest Classifier** with anti-overfitting measures  
✔ **Production-Ready Web Interface** built with Flask  
✔ **Model Persistence** using joblib serialization  
✔ **Customizable Training** via CLI arguments  
✔ **Explainable AI** with confidence scores  

</div>

<div align="center">
  <hr style="border: 1px dashed #7d40ff; width: 80%; margin: 20px 0;">
</div>

## 📊 Performance Metrics

<div align="center" style="margin: 20px auto; max-width: 600px;">

| Metric        | Score   | Improvement | Visual |
|--------------|---------|-------------|--------|
| Accuracy     | 92.4%   | +7.2%       | ▰▰▰▰▰▰▰▰▰▰ |
| Precision    | 91.8%   | +6.5%       | ▰▰▰▰▰▰▰▰▰  |
| Recall       | 93.1%   | +8.1%       | ▰▰▰▰▰▰▰▰▰▰ |
| F1-Score     | 92.4%   | +7.3%       | ▰▰▰▰▰▰▰▰▰▰ |

</div>


<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=wave&color=7d40ff&height=30&section=divider"/>
</div>


## 🛠️ Installation Guide

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

<div align="center"> <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=30&section=divider"/> </div>

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


<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/240304586-218f11fa-42f4-4af1-9e0a-a2a8e8a7f4e9.gif" width="100%" height="2px"/>
</div>

🚀 Usage
Training the Model
python fake_news_detector.py \
  --true datasets/True.csv \
  --fake datasets/Fake.csv \
  --test_size 0.2 \
  --random_state 42

Running the Web Interface
flask run --host=0.0.0.0 --port=5000



<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/240304586-218f11fa-42f4-4af1-9e0a-a2a8e8a7f4e9.gif" width="100%" height="2px"/>
</div>

📂 Project Structure
<div style = "align=center">
fake-news-detector/
├── app.py                # Flask application
├── fake_news_detector.py # Core training script
├── interface.py          # Prediction logic
├── requirements.txt      # Dependencies
├── datasets/
│   ├── True.csv          # Real news samples
│   └── Fake.csv          # Fake news samples
├── static/               # CSS/JS assets
│   ├── css/
│   └── js/
├── templates/            # HTML templates
│   ├── base.html
│   └── index.html
└── models/               # Saved models
    ├── fake_news_model.pkl
    └── vectorizer.pkl
    </div>



<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/240304586-218f11fa-42f4-4af1-9e0a-a2a8e8a7f4e9.gif" width="100%" height="2px"/>
</div>


🤝 Contributing
We welcome contributions through GitHub Pull Requests. Please:

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some amazing feature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request


<div align="center"> <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=30&section=divider"/> </div>

✉️ Contact
Muhammad Abuhurera
📧 abuhurerarchani@gmail.com
🌐 GitHub Profile





