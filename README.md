# 🐦 Tweet Sentiment Analyzer

A sentiment analysis project trained on the **Sentiment140** dataset (1.6 million tweets), with a Streamlit web app for real-time predictions.

## Overview

This project classifies tweets as **Positive** or **Negative** using classical machine learning. It covers the full pipeline from raw data to a deployed, interactive app.

## Project Pipeline

1. **Data Preprocessing** — cleaned raw tweets (removed URLs, @mentions, hashtags symbols, non-alphabetic characters, extra whitespace, lowercased text).
2. **Exploratory Data Analysis (EDA)** — class distribution, tweet length distribution, most common words by sentiment, word clouds.
3. **Feature Engineering** — text converted to numeric vectors using **TF-IDF** (unigrams + bigrams, 50,000 max features).
4. **Model Development** — trained and compared three models on the same train/test split:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Random Forest (trained on a stratified subsample for tractability)
5. **Model Selection** — best model chosen by F1 score, saved with `joblib` along with the fitted TF-IDF vectorizer.
6. **Deployment** — a Streamlit app (`app.py`) loads the saved model and vectorizer to predict sentiment on any user-entered text in real time.

## Files

| File | Description |
|---|---|
| `Sentiment_Analysis_Sentiment140.ipynb` | Full notebook: EDA, preprocessing, feature engineering, model training/comparison, and model saving. |
| `app.py` | Streamlit app that loads the saved model/vectorizer and serves predictions through a simple UI. |
| `sentiment_model.joblib` | The best-performing trained model (saved from the notebook). |
| `tfidf_vectorizer.joblib` | The fitted TF-IDF vectorizer used to transform raw text into model input. |
| `requirements.txt` | Python dependencies needed to run the app. |

> Note: the raw dataset (`training.1600000.processed.noemoticon.csv`, ~228MB) is **not included** in this repo due to its size. It can be downloaded from [Kaggle's Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) if you want to re-run the notebook from scratch.

## Running Locally

1. Clone this repository and navigate into the project folder.
2. Create and activate a virtual environment (conda or venv), e.g.:
   ```bash
   conda create -n sentiment python=3.11 -y
   conda activate sentiment
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. Open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

## How It Works

The app cleans input text with the same function used during training, transforms it with the saved TF-IDF vectorizer, and feeds it to the saved model for a Positive/Negative prediction (with a confidence score, if the model supports `predict_proba`).

## Tech Stack

- **Python**
- **scikit-learn** — TF-IDF vectorization, model training
- **pandas / numpy** — data handling
- **matplotlib / seaborn / wordcloud** — EDA visualizations
- **Streamlit** — web app interface
- **joblib** — model persistence

## Possible Future Enhancements

- Compare against deep learning approaches (e.g. a fine-tuned DistilBERT model).
- Add a neutral sentiment class using additional labeled data.
- Deploy with a REST API (e.g. FastAPI) in addition to the Streamlit UI.
- Add batch prediction support (upload a CSV of tweets, get sentiment for all of them).

## Dataset Citation

Go, A., Bhayani, R. and Huang, L., 2009. *Twitter sentiment classification using distant supervision.* CS224N project report, Stanford, 1(2009), p.12.
