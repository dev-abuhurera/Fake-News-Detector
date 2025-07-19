# 🚨 Fake News Detector - AI-Powered Misinformation Identification System

![Project Banner](https://github.com/user-attachments/assets/32c44c73-e6f8-42be-b2df-3b1e03dcca7a)

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

## 📊 Performance Metrics

<div align="center" style="margin: 20px auto; max-width: 600px;">

| Metric        | Score   | Improvement | Visual |
|--------------|---------|-------------|--------|
| Accuracy     | 92.4%   | +7.2%       | ▰▰▰▰▰▰▰▰▰▰ |
| Precision    | 91.8%   | +6.5%       | ▰▰▰▰▰▰▰▰▰  |
| Recall       | 93.1%   | +8.1%       | ▰▰▰▰▰▰▰▰▰▰ |
| F1-Score     | 92.4%   | +7.3%       | ▰▰▰▰▰▰▰▰▰▰ |

</div>

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
python fake_news_detector.py \
  --true datasets/True.csv \
  --fake datasets/Fake.csv \
  --test_size 0.2 \
  --random_state 42

Running the Web Interface
flask run --host=0.0.0.0 --port=5000



🖥️ Interface Preview
<div align="center" style="display: flex; flex-wrap: wrap; gap: 15px; justify-content: center;"> <img src="https://github.com/user-attachments/assets/32c44c73-e6f8-42be-b2df-3b1e03dcca7a" width="45%" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(125, 64, 255, 0.2);"/> <img src="https://github.com/user-attachments/assets/b2e88c02-145d-40ee-b7d0-65a48548cf75" width="45%" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(125, 64, 255, 0.2);"/> </div>



