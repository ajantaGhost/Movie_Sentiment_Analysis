🎬 Sentiment Analysis Movie Review App

A Movie Review Sentiment Analysis project built in Python. This app predicts whether a movie review or any text is Positive 👍 or Negative 👎 using Machine Learning and NLP techniques.

Demo

Features

Preprocess movie reviews: clean text, remove stopwords, apply stemming

TF-IDF vectorization for text to numeric conversion

Logistic Regression model for classification

Interactive web app using Streamlit

Real-time sentiment prediction

Project Structure
SentimentAnalysisProjectByMe/
│
├─ IMDB_Dataset.xlsx        # Movie review dataset
├─ sentiment_training.py    # Script to train the model
├─ app.py                   # Streamlit app for prediction
├─ sentiment_model.pkl      # Trained model (generated after training)
├─ vectorizer.pkl           # TF-IDF vectorizer (generated after training)
├─ label_encoder.pkl        # Label encoder (generated after training)
└─ README.md                # Project documentation

Dataset

IMDB_Dataset.xlsx contains:

text: Movie review text

sentiment: Label (positive or negative)

Installation

Clone this repository:

git clone https://github.com/yourusername/SentimentAnalysisProject.git
cd SentimentAnalysisProject


Install required Python packages:

pip install pandas numpy scikit-learn nltk streamlit openpyxl


Download NLTK stopwords:

import nltk
nltk.download('stopwords')

Usage
1. Train the Model

Run the training script to preprocess the data and save the trained model:

python sentiment_training.py


Generates:

sentiment_model.pkl

vectorizer.pkl

label_encoder.pkl

2. Run the Streamlit App
python -m streamlit run app.py


Opens a browser window.

Enter a movie review or text.

Click Predict Sentiment to see the result.

How It Works

Text Preprocessing:

Remove HTML tags and special characters

Convert text to lowercase

Remove stopwords

Apply stemming

Vectorization:

Convert text into numeric form using TF-IDF

Model:

Logistic Regression classifies text as positive or negative

Prediction:

User input is preprocessed, vectorized, and fed to the trained model

Sentiment result is displayed

Dependencies

Python 3.10+

Pandas

NumPy

Scikit-learn

NLTK

Streamlit

Openpyxl

Author

Ajanta Ghosh

GitHub: https://github.com/yourusername

License

This project is for educational purposes.
