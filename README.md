# AI Phishing Email Detection 

## Description
Detect phishing emails using classical ML and lightweight NLP embeddings using the dataset from Kaggle. 

## Features
- End-to-end pipeline: ML is trained on the dataset from Kaggle, evaluates phishing email vs legit email, and saves artifacts. 

- Reproducible experiments in Jupyter notebooks under notebooks/

- Model artifacts saved to models/ and metrics/plots to results/.

- Configurable to work with your own labeled email data (CSV).

## Installation
git clone https://github.com/binaabdulrahim/aiPhishingEmailDetectionProject.git
cd aiPhishingEmailDetectionProject

# Python 3.10+ recommended
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

```
## Usage
```bash
# Run the project
npm start  # or python main.py
```
python -m pip install jupyter
jupyter lab   # or: jupyter notebook

## Configuration
Input CSV columns: text, label

Vectorization: TF-IDF (word/char n-grams) or embeddings (optional)

Model: Start with Logistic Regression/SVM baseline; compare others.

Reproducibility: set random_state in splits/models.



