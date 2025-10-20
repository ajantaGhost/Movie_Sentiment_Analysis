import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords (first time only)
nltk.download('stopwords')

# Load trained model, vectorizer, and label encoder
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to preprocess input text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # remove non-alphabetic chars
    text = text.lower()
    words = [ps.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Function to predict sentiment
def predict_sentiment(text):
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)
    return le.inverse_transform(pred)[0]

# Streamlit App
st.title("Sentiment Analysis App")
st.write("Enter a movie review or text, and find out if it's Positive or Negative!")

# Input text
user_input = st.text_area("Enter your text here:")

# Prediction button
if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        sentiment = predict_sentiment(user_input)
        if sentiment == 'positive':
            st.success(f"Sentiment: {sentiment.capitalize()} 👍")
        else:
            st.error(f"Sentiment: {sentiment.capitalize()} 👎")
    else:
        st.warning("Please enter some text to predict.")