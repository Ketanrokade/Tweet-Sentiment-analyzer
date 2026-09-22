import re
import joblib
import streamlit as st

# ---- Load the saved model + vectorizer (trained in the notebook) ----
@st.cache_resource
def load_artifacts():
    model = joblib.load("sentiment_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_artifacts()

# ---- Same cleaning function used during training ----
def clean_tweet(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def predict_sentiment(text):
    cleaned = clean_tweet(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    label = "Positive" if pred == 4 else "Negative"
    if hasattr(model, "predict_proba"):
        conf = model.predict_proba(vec)[0].max()
        return label, conf
    return label, None

# ---- UI ----
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="🐦")
st.title("🐦 Tweet Sentiment Analyzer")
st.write("Type a tweet below and see the predicted sentiment in real time.")

user_input = st.text_area("Enter a tweet:", height=100,
                           placeholder="e.g. Just had the best coffee ever!")

if st.button("Predict Sentiment") and user_input.strip():
    label, conf = predict_sentiment(user_input)
    if label == "Positive":
        st.success(f"Prediction: **{label}**" + (f"  (confidence: {conf:.1%})" if conf else ""))
    else:
        st.error(f"Prediction: **{label}**" + (f"  (confidence: {conf:.1%})" if conf else ""))

st.markdown("---")
st.caption("Model: Logistic Regression + TF-IDF, trained on the Sentiment140 dataset (1.6M tweets).")
