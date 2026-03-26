import streamlit as st
import joblib
import re
import numpy as np

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# -----------------------------
# HIGHLIGHT KEYWORDS
# -----------------------------
def highlight_words(text, words):
    for word in words:
        text = re.sub(f"({word})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    return text

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="💬", layout="centered")

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 style='text-align:center;'>💬 AI Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>🚀 Advanced NLP Sentiment Detection System</p>", unsafe_allow_html=True)

# -----------------------------
# INPUT
# -----------------------------

if "text" not in st.session_state:
    st.session_state.text = ""

user_input = st.text_area("✍️ Enter your text", value=st.session_state.text,height=150)

col1, col2 = st.columns(2)

# -----------------------------
# 👉 SINGLE BUTTON (IMPORTANT)
# -----------------------------
if st.button("🔍 Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")

    else:
        # Clean + Transform
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])

        # Prediction
        prediction = model.predict(vector)[0]
        label = le.inverse_transform([prediction])[0]

        # Confidence
        try:
            score = model.decision_function(vector)[0]
            confidence = float(np.max(score))
            confidence = round(abs(confidence), 2)
        except:
            confidence = 0.5

        confidence_percent = int(min(max(confidence * 20, 0), 100))

        # Emoji mapping
        emoji = {
            "positive": "😊",
            "neutral": "😐",
            "negative": "😠"
        }

        # -----------------------------
        # RESULT
        # -----------------------------
        st.subheader("📊 Result")

        if label.lower() == "positive":
            st.success(f"{emoji['positive']} {label} Sentiment")

        elif label.lower() == "neutral":
            st.warning(f"{emoji['neutral']} {label} Sentiment")

        else:
            st.error(f"{emoji['negative']} {label} Sentiment")

        # -----------------------------
        # CONFIDENCE
        # -----------------------------
        st.markdown("### 🎯 Confidence Meter")
        st.progress(confidence_percent)
        st.write(f"Confidence: **{confidence_percent}%**")

        # -----------------------------
        # KEYWORDS
        # -----------------------------
        important_words = ["good", "bad", "excellent", "worst", "amazing", "poor"]
        highlighted_text = highlight_words(user_input, important_words)

        st.markdown("### 🧠 Key Influencing Words")
        st.markdown(highlighted_text, unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 Project Info")

st.sidebar.info("""
🔹 Model: Linear SVM  
🔹 Vectorizer: TF-IDF  
🔹 Accuracy: ~79%  
""")


