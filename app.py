import streamlit as st
import joblib
import re
import numpy as np
import time
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="💬", layout="centered")

# -----------------------------
# 🎨 GLASSMORPHISM UI

st.markdown("""
<style>
/* 🌙 Ultra Soft Dark Background */
.stApp {
    background: linear-gradient(135deg, #0b0f1a, #111827, #1f2937);
    color: #e5e7eb;
}



/* Title */
h1 {
    text-align: center;
    color: #60a5fa;
}

/* Buttons (softer blue) */
.stButton>button {
    border-radius: 10px;
    background: #2563eb;
    color: white;
    font-weight: 600;
}

.stButton>button:hover {
    background: #1d4ed8;
}

/* Text area */
textarea {
    border-radius: 10px !important;
    background-color: #111827 !important;
    color: #e5e7eb !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
}
</style>
""", unsafe_allow_html=True)
# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1>💬 AI Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyze phone reviews with AI 🚀</p>", unsafe_allow_html=True)


# -----------------------------
# INPUT CARD
# -----------------------------
if "text" not in st.session_state:
    st.session_state.text = ""

text = st.text_area("Enter Phone Review:", value=st.session_state.text)

st.markdown('<div class="glass">', unsafe_allow_html=True)

if "text" not in st.session_state:
    st.session_state.text = ""

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("😊 Positive"):
        st.session_state.text = "This phone is amazing! Battery life is great and camera quality is excellent."

with col2:
    if st.button("😠 Negative"):
        st.session_state.text = "Worst phone ever. It hangs a lot and battery drains quickly."

with col3:
    if st.button("😐 Neutral"):
        st.session_state.text = "The phone is okay, average performance and decent features."


analyze = st.button("🔍 Analyze Sentiment")

st.markdown('</div>', unsafe_allow_html=True)
# -----------------------------

# LOAD MODEL
# -----------------------------
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# TEXT CLEANING FUNCTION
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

# -----------------------------
# PREDICTION
# -----------------------------
if analyze:

    if text.strip() == "":
        st.warning("⚠️ Please enter some text")

    else:
        # ⏳ LOADER
        with st.spinner("Analyzing sentiment..."):
            time.sleep(1.5)

        cleaned = clean_text(text)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]
        label = le.inverse_transform([prediction])[0]

        try:
            score = model.decision_function(vector)[0]
            confidence = float(np.max(score))
            confidence = round(abs(confidence), 2)
        except:
            confidence = 0.5

        confidence_percent = int(min(max(confidence * 20, 0), 100))

        emoji = {
            "positive": "😊",
            "neutral": "😐",
            "negative": "😠"
        }

        # -----------------------------
        # RESULT CARD
        # -----------------------------
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.markdown("## 📊 Prediction Result")

        if label.lower() == "positive":
            st.success(f"{emoji['positive']} {label} Sentiment")
        elif label.lower() == "neutral":
            st.warning(f"{emoji['neutral']} {label} Sentiment")
        else:
            st.error(f"{emoji['negative']} {label} Sentiment")

        st.markdown("### 🎯 Confidence Meter")
        st.progress(confidence_percent)
        st.write(f"Confidence: **{confidence_percent}%**")

        # -----------------------------
     
        # -----------------------------
        # KEYWORDS
        # -----------------------------
        important_words = ["good", "bad", "excellent", "worst", "amazing", "poor"]
        highlighted_text = highlight_words(text, important_words)

        st.markdown("### 🧠 Key Influencing Words")
        st.markdown(highlighted_text, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 Project Info")

st.sidebar.info("""
🔹 Model: Linear SVM  
🔹 Vectorizer: TF-IDF   """)

