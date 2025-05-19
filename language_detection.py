# language_detection.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('language_dataset.csv')  # Replace with your dataset path
print("Sample Data:")
print(data.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Language'], test_size=0.2, random_state=42)

# Build pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
