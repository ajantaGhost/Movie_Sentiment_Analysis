import re
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import nltk

# Download stopwords
nltk.download('stopwords')

print("Step 1:...")

# Load dataset
dataset = pd.read_excel(
    r"D:\Ajanta_Ghosh\VS-code\SentimentAnalysisMiniProject\IMDB_Dataset.xlsx",
    usecols=['text', 'sentiment']  # only load these columns
)


print("Step 2:...")

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text cleaning and stemming
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # remove non-alphabetic chars
    text = text.lower()
    words = [ps.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

dataset['clean_text'] = dataset['text'].apply(preprocess_text)


print("Step 3:...")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(dataset['sentiment'])  # positive -> 1, negative -> 0


print("Step 4:...")

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(dataset['clean_text']).toarray()


print("Step 5:...")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Step 6:...")

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


print("Step 7:...")

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


print("Step 8:...")

# Save model, vectorizer, and label encoder
pickle.dump(model, open('sentiment_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))

print("Training completed and files saved successfully!")
