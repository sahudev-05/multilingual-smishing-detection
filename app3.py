import re
import numpy as np
import streamlit as st
import torch
import xgboost as xgb
import xgboost
print(xgboost.__version__)

from transformers import AutoTokenizer, AutoModel
from langdetect import detect



# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Multilingual Smishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#f5f7fb;
}

h1,h2,h3{
    color:#0E4C92;
}

.stButton>button{
    width:100%;
    height:3.2em;
    border-radius:10px;
    background:linear-gradient(90deg,#0E4C92,#1976D2);
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1565C0;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD XGBOOST
# ==========================================================



@st.cache_resource
def load_xgb():
    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_multilingual.json")
    return model


@st.cache_resource
def load_xlmr():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModel.from_pretrained("xlm-roberta-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device




with st.spinner("Loading AI Models..."):

    xgb_model = load_xgb()


    tokenizer, bert_model, device = load_xlmr()


# ==========================================================
# EMBEDDING
# ==========================================================

def get_embedding(text):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()
    }

    with torch.inference_mode():
        outputs = bert_model(**encoded)

    embedding = outputs.last_hidden_state[:, 0, :]

    return embedding.cpu().numpy()
# ==========================================================
# LANGUAGE
# ==========================================================

def detect_language(text):

    try:
        return detect(text)
    except:
        return "Unknown"


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🛡️ Smishing Detection")

st.sidebar.markdown("---")

st.sidebar.info("""
This system detects multilingual SMS phishing attacks using:

• XLM-RoBERTa

• XGBoost

• Language Detection

• Rule-based Explanation
""")

st.sidebar.markdown("---")



st.sidebar.subheader("Sample Messages")

legitimate = """Hi Rahul,

I'll reach the office around 10 AM.

Let's have lunch together.
"""

smishing = """Your SBI account has been suspended.

Verify immediately at

http://fakebank.com
"""

choice = st.sidebar.selectbox(
    "Choose Example",
    [
        "None",
        "Legitimate",
        "Smishing"
    ]
)

if choice=="Legitimate":
    default_text = legitimate
elif choice=="Smishing":
    default_text = smishing
else:
    default_text = ""

# ==========================================================
# MAIN PAGE
# ==========================================================

st.title("📱 Multilingual Smishing Detection System")

st.write("""
Detect multilingual SMS phishing attacks using
**XLM-RoBERTa embeddings** and an **XGBoost classifier**.
""")

message = st.text_area(
    "Enter SMS Message",
    value=default_text,
    height=220,
    placeholder="Paste your SMS here..."
)
# ==========================================================
# SUSPICIOUS KEYWORDS
# ==========================================================

SUSPICIOUS_KEYWORDS = [

    "otp",
    "verify",
    "verification",
    "password",
    "login",
    "bank",
    "account",
    "kyc",
    "upi",
    "wallet",
    "credit",
    "debit",
    "payment",
    "refund",
    "click",
    "link",
    "urgent",
    "immediately",
    "update",
    "confirm",
    "security",
    "blocked",
    "suspended",
    "expired",
    "alert",
    "reward",
    "winner",
    "gift",
    "claim",
    "offer",
    "free",
    "prize",
    "cash"

]

# ==========================================================
# HIGHLIGHT MESSAGE
# ==========================================================
def highlight_text(text):
    # Escape HTML first
    import html
    highlighted = html.escape(text)

    # Highlight URLs first
    highlighted = re.sub(
        r"(https?://\S+|www\.\S+)",
        r"<span style='color:red;font-weight:bold;'>\1</span>",
        highlighted,
        flags=re.IGNORECASE
    )

    # Highlight suspicious words
    for word in SUSPICIOUS_KEYWORDS:
        highlighted = re.sub(
            rf"\b({re.escape(word)})\b",
            r"<span style='color:red;font-weight:bold;'>\1</span>",
            highlighted,
            flags=re.IGNORECASE
        )

    # Preserve line breaks
    highlighted = highlighted.replace("\n", "<br>")

    return highlighted


# ==========================================================
# EXPLANATION ENGINE
# ==========================================================

def explain_prediction(text):

    reasons = []

    if re.search(r"https?://|www\.", text, re.IGNORECASE):
        reasons.append("🔗 Contains a URL that could redirect to a phishing website.")

    if re.search(r"\botp\b", text, re.IGNORECASE):
        reasons.append("🔐 Mentions an OTP or authentication code.")

    if re.search(r"verify|verification|confirm|login|password|kyc", text, re.IGNORECASE):
        reasons.append("👤 Requests identity verification or sensitive credentials.")

    if re.search(r"bank|account|wallet|upi|credit|debit|payment|refund", text, re.IGNORECASE):
        reasons.append("🏦 Contains banking or financial terms.")

    if re.search(r"urgent|immediately|blocked|suspended|expired|alert", text, re.IGNORECASE):
        reasons.append("⚠ Uses urgency or fear tactics.")

    if re.search(r"winner|reward|gift|claim|offer|free|prize", text, re.IGNORECASE):
        reasons.append("🎁 Offers rewards or prizes commonly seen in scams.")

    if len(text.split()) < 5:
        reasons.append("📝 Very short messages can sometimes be suspicious.")

    return reasons


# ==========================================================
# RISK LEVEL
# ==========================================================

def get_risk_level(smishing_prob):

    if smishing_prob >= 80:
        return "🔴 HIGH"

    elif smishing_prob >= 50:
        return "🟠 MEDIUM"

    else:
        return "🟢 LOW"


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_sms(message):

    embedding = get_embedding(message)

    prediction = xgb_model.predict(embedding)[0]

    probability = xgb_model.predict_proba(embedding)[0]

    confidence = float(np.max(probability)) * 100

    language = detect_language(message)


    

    # Class Mapping
    # 0 = Smishing
    # 1 = Legitimate

    smishing_prob = float(probability[0]) * 100
    legitimate_prob = float(probability[1]) * 100

    if prediction == 0:
        risk = get_risk_level(smishing_prob)
    else:
        risk = "🟢 LOW"

    return {

        "prediction": prediction,

        "probability": probability,

        "confidence": confidence,

        "language": language,

        "risk": risk,

        "smishing_prob": smishing_prob,

        "legitimate_prob": legitimate_prob

    }


# ==========================================================
# SHOW EXPLANATION
# ==========================================================

def show_explanation(message):

    st.subheader("💡 Why was this classified?")

    reasons = explain_prediction(message)

    if reasons:

        for reason in reasons:
            st.write(reason)

    else:

        st.success(
            "No obvious phishing indicators were detected."
        )


# ==========================================================
# SHOW HIGHLIGHTED MESSAGE
# ==========================================================
def show_highlighted_message(message):

    st.subheader("🔍 Highlighted Message")

    st.markdown(
        f"""
        <div style="
            background:#000000;
            color:white;
            padding:20px;
            border-radius:12px;
            border:2px solid #444;
            font-size:17px;
            line-height:1.8;
            white-space:pre-wrap;
            font-family:Arial,sans-serif;
        ">
        {highlight_text(message)}
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================================
# TECHNICAL DETAILS
# ==========================================================

def show_technical_details(result):

    with st.expander("⚙ Technical Details"):

        st.write(f"**Detected Language:** {result['language']}")

        st.write(f"**Confidence:** {result['confidence']:.2f}%")

        st.write(f"**Risk Level:** {result['risk']}")

        st.write(f"**Embedding Size:** 768")

        st.write(f"**Device:** {device}")

        st.write("**Embedding Model:** XLM-RoBERTa Base")

        st.write("**Classifier:** XGBoost")

        st.write("**Class Mapping:**")

        st.code(
            "0 = Smishing\n1 = Legitimate"
        )
# ==========================================================
# PREDICT BUTTON
# ==========================================================

if st.button("🚀 Analyze Message"):

    if message.strip() == "":
        st.warning("Please enter an SMS message.")
        st.stop()

    with st.spinner("Analyzing message..."):
        result = predict_sms(message)

    prediction = result["prediction"]
    confidence = result["confidence"]
    language = result["language"]
    risk = result["risk"]

    smishing_prob = result["smishing_prob"]
    legitimate_prob = result["legitimate_prob"]

    st.divider()
    

    # ======================================================
    # RESULT
    # ======================================================

    if prediction == 0:

        st.error("🚨 Smishing Message Detected")

        result_color = "#ff0318"

    else:

        st.success("✅ Legitimate Message")

        result_color = "#00cf30"

    st.markdown(
        f"""
        <div style="
        background:{result_color};
        padding:20px;
        border-radius:15px;
        border-left:8px solid #1976D2;
        font-size:20px;
        font-weight:bold;">

        Prediction Completed Successfully

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ======================================================
    # METRICS
    # ======================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with col2:
        st.metric(
            "Language",
            language.upper()
        )

    with col3:
        st.metric(
            "Risk Level",
            risk
        )

    st.divider()

    # ======================================================
    # CONFIDENCE BAR
    # ======================================================

    st.subheader("🎯 Model Confidence")

    st.progress(confidence / 100)

    st.write(f"Model Confidence: **{confidence:.2f}%**")

    st.divider()

    # ======================================================
    # PROBABILITIES
    # ======================================================

    st.subheader("📊 Prediction Probabilities")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "🚨 Smishing",
            f"{smishing_prob:.2f}%"
        )

        st.progress(smishing_prob / 100)

    with c2:

        st.metric(
            "✅ Legitimate",
            f"{legitimate_prob:.2f}%"
        )

        st.progress(legitimate_prob / 100)

    st.divider()

    # ======================================================
    # MESSAGE PREVIEW
    # ======================================================

    show_highlighted_message(message)

    st.divider()

    # ======================================================
    # EXPLANATION
    # ======================================================

    if prediction == 0:

        show_explanation(message)

    else:

        st.subheader("💡 Explanation")

        st.success(
            """
No significant phishing indicators were detected.

The message appears conversational or informational.
            """
        )

    st.divider()

    # ======================================================
    # TECHNICAL DETAILS
    # ======================================================

    show_technical_details(result)

# ==========================================================
# SUPPORTED LANGUAGES
# ==========================================================

SUPPORTED_LANGUAGES = [
    "English",
    "Hindi",
    "Bengali",
    "Marathi",
    "Punjabi",
    "Urdu",
    "Arabic",
    "Chinese",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
    "Ukrainian",
    "Turkish",
    "Indonesian",
    "Javanese",
    "Swedish",
    "Norwegian"
]

st.divider()

st.subheader("🌍 Supported Languages")

st.info(
    "This multilingual smishing detection system currently supports the following languages:"
)

cols = st.columns(3)

for i, lang in enumerate(SUPPORTED_LANGUAGES):
    cols[i % 3].write(f"• {lang}")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Safety Tips")

st.sidebar.info(
"""
✔ Never share your OTP.

✔ Never reveal passwords.

✔ Verify banking messages from official websites.

✔ Avoid clicking unknown links.

✔ Ignore lottery or prize messages.

✔ Check URLs carefully before entering credentials.
"""
)

st.sidebar.markdown("---")

st.sidebar.success(
"""
### 🤖 Models Used

• XLM-RoBERTa Base

• XGBoost Classifier

• Language Detection

• Rule-based Explanation Engine
"""
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div class='footer'>

<h3>📱 Multilingual Smishing Detection System</h3>

Developed using
<b>XLM-RoBERTa</b> embeddings and an
<b>XGBoost Classifier</b> for multilingual SMS phishing detection.

<br><br>

Made for <b>NLP Project</b>

</div>
""",
unsafe_allow_html=True
)