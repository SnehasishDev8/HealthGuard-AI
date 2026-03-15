# train_model.py
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob

# Load your data
df = pd.read_csv("data/symptom_specialty_data.csv")

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = str(TextBlob(text).correct())  # Spell check
    return text.strip()

df['cleaned'] = df['Symptom'].apply(clean_text)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['Specialty']

model = MultinomialNB()
model.fit(X, y)

# Save model
pickle.dump(model, open("models/doctor_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("✅ Model trained and saved.")
