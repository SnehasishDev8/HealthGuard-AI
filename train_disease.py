import pandas as pd
import re
import pickle
import logging
import os
from pathlib import Path
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiseaseModelTrainer:
    def __init__(self, data_path="data/decease.csv", models_dir="models"):
        self.data_path = data_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        self.df = None

    def load_and_validate_data(self):
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)

            required_columns = ['Symptoms', 'Disease']
            for col in required_columns:
                if col not in self.df.columns:
                    raise ValueError(f"Missing column: {col}")

            before = len(self.df)
            self.df = self.df.dropna(subset=required_columns).drop_duplicates()
            after = len(self.df)

            logger.info(f"Data loaded: {before} -> {after} records after cleaning")
            logger.info(f"Unique diseases: {self.df['Disease'].nunique()}")
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def clean_symptom_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r"[\[\]'\"()]", "", text)
        text = re.sub(r"[,;]+", " ", text)
        text = re.sub(r"\s+", " ", text).lower().strip()
        try:
            text = str(TextBlob(text).correct())
        except Exception as e:
            logger.warning(f"Spell correction failed: {text[:30]} - {e}")
        return text

    def preprocess_data(self):
        logger.info("Preprocessing data...")
        self.df["cleaned_symptoms"] = self.df["Symptoms"].apply(self.clean_symptom_text)
        self.df = self.df[self.df["cleaned_symptoms"].str.len() > 0]
        self.label_encoder = LabelEncoder()
        self.df["disease_encoded"] = self.label_encoder.fit_transform(self.df["Disease"])
        logger.info(f"Preprocessing complete. Final dataset size: {len(self.df)}")

    def train_model(self, test_size=0.2, random_state=42):
        logger.info("Training model...")

        X_text = self.df["cleaned_symptoms"]
        y = self.df["Disease"]

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y, test_size=test_size, random_state=random_state
        )

        self.vectorizer = TfidfVectorizer(
            max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=1, max_df=0.95
        )
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)

        self.model = MultinomialNB(alpha=0.1)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")

        min_class_count = min(Counter(y_train).values())
        safe_cv = min(5, min_class_count)

        if safe_cv < 2:
            logger.warning("Not enough data per class for cross-validation. Skipping.")
            cv_scores = []
        else:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=safe_cv)

        logger.info(f"Cross-validation scores: {cv_scores}")
        if len(cv_scores) > 0:
          logger.info(f"Mean CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

            

        logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
        return accuracy

    def save_models(self):
        try:
            joblib.dump(self.model, self.models_dir / "disease_model.pkl")
            joblib.dump(self.vectorizer, self.models_dir / "disease_vectorizer.pkl")
            joblib.dump(self.label_encoder, self.models_dir / "label_encoder.pkl")

            metadata = {
                'model_type': 'MultinomialNB',
                'vectorizer_type': 'TfidfVectorizer',
                'feature_count': self.vectorizer.max_features,
                'disease_classes': list(self.label_encoder.classes_),
                'training_samples': len(self.df)
            }
            with open(self.models_dir / "model_metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"Models saved in {self.models_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False

    def generate_training_report(self):
        report_path = self.models_dir / "training_report.txt"
        with open(report_path, "w") as f:
            f.write("=== Disease Prediction Model Report ===\n\n")
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Samples: {len(self.df)}\n")
            f.write(f"Unique diseases: {self.df['Disease'].nunique()}\n")
            f.write(f"Avg symptom length: {self.df['cleaned_symptoms'].str.split().str.len().mean():.2f}\n")
            f.write("\nTop diseases:\n")
            f.write(str(self.df["Disease"].value_counts().head(10)))

        logger.info(f"Training report saved to {report_path}")

def main():
    trainer = DiseaseModelTrainer()
    if not trainer.load_and_validate_data():
        return

    trainer.preprocess_data()
    accuracy = trainer.train_model()

    if accuracy > 0.4:
        if trainer.save_models():
            trainer.generate_training_report()
            logger.info("✅ Training completed successfully.")
        else:
            logger.error("❌ Model saving failed.")
    else:
        logger.warning(f"⚠️ Accuracy too low: {accuracy:.4f}. Improve your dataset.")

if __name__ == "__main__":
    main()
