
import streamlit as st
import tensorflow as tf
import pickle
import string
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#  Load model and tokenizer 
model = tf.keras.models.load_model('spam_model.keras')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 100

#  Preprocessing functions 
def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    words = text.lower().split()
    return ' '.join([w for w in words if w not in stop_words])

def preprocess_email(text):
    text = text.replace('Subject', '')
    text = remove_punctuations(text)
    text = remove_stopwords(text)
    return text

def predict_spam(email_text):
    processed = preprocess_email(email_text)
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    label = "Spam" if prediction > 0.5 else "Not Spam"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

#  Streamlit UI 
st.set_page_config(page_title="Spam Email Classifier", page_icon="📩")
st.title(" Spam Email Classifier")
# Example emails
spam_examples = [
    "Subject: Congratulations! You've won a free iPhone 15. Click the link to claim now!",
    "Subject: Get rich quick! Invest $500 in our crypto scheme and earn $10,000 in a week."
]

ham_examples = [
    "Subject: Meeting Reminder\nDon't forget about our team meeting tomorrow at 10 AM in the conference room.",
    "Subject: Dinner Plans\nAre we still on for dinner tonight? Let me know what time works for you."
]

# Section for sample email buttons
st.subheader("📌 Try Sample Emails")

col1, col2 = st.columns(2)

with col1:
    if st.button("Spam Example 1"):
        email_input = spam_examples[0]
    if st.button("Spam Example 2"):
        email_input = spam_examples[1]

with col2:
    if st.button("Non-Spam Example 1"):
        email_input = ham_examples[0]
    if st.button("Non-Spam Example 2"):
        email_input = ham_examples[1]

# Persistent email input
if "email_input" not in st.session_state:
    st.session_state.email_input = ""

email_input = st.text_area("Paste your email content below:", value=st.session_state.email_input, height=200)

# Update session state when a button is clicked
if st.button("Check Spam"):
    if email_input.strip() == "":
        st.warning("Please enter some email content first.")
    else:
        st.session_state.email_input = email_input
        label, confidence = predict_spam(email_input)
        if label == "Spam":
            st.error(f" Prediction: **{label}** ({confidence:.2%} confidence)")
        else:
            st.success(f" Prediction: **{label}** ({confidence:.2%} confidence)")
