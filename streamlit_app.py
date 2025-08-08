
import streamlit as st
import tensorflow as tf
import pickle
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#  Load model and tokenizer 
model = tf.keras.models.load_model('spam_model.h5')

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

email_input = st.text_area("Paste your email content below:", height=200)

if st.button("Check Spam"):
    if email_input.strip() == "":
        st.warning("Please enter some email content first.")
    else:
        label, confidence = predict_spam(email_input)
        if label == "Spam":
            st.error(f" Prediction: **{label}** ({confidence:.2%} confidence)")
        else:
            st.success(f" Prediction: **{label}** ({confidence:.2%} confidence)")
