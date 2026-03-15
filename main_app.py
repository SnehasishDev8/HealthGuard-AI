from google import genai
import json
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd
import logging
import os
from pathlib import Path
from textblob import TextBlob
from datetime import datetime
import re
from werkzeug.exceptions import BadRequest

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY environment variable is not set. API calls will fail.")

try:
    gemini_client = genai.Client(api_key=api_key) if api_key else genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    gemini_client = None
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

from config import Config
app.secret_key = Config.SECRET_KEY
app.config["DEBUG"] = Config.DEBUG


class DiseasePredictor:
    def __init__(self, *args, **kwargs):
        self.model_metadata = {
            "model_type": "Gemini 2.5 Flash API",
            "description": "Predicts diseases and medical info dynamically using Gemini."
        }
        logger.info("✅ Gemini Disease Predictor initialized")

    def predict_disease(self, symptoms):
        """Predict disease and get medical info using Gemini AI"""
        prompt = f"""
        You are an AI medical assistant. A user has reported the following symptoms: "{symptoms}".
        Analyze these symptoms and provide the most likely primary disease, a confidence score (between 0.0 and 1.0), and up to 2 alternative diseases with their confidence scores.
        Also provide medical information for the primary disease: treatment, medicinal composition, ingredients to avoid, recommended diet, and precautionary measures.
        
        Return ONLY a valid JSON object with the following structure, with no extra text or markdown:
        {{
            "primary_disease": "string",
            "confidence": 0.0,
            "alternative_predictions": [
                {{"disease": "string", "confidence": 0.0}}
            ],
            "medical_info": {{
                "treatment": "string",
                "medicinal_composition": "string",
                "ingredients_to_avoid": "string",
                "recommended_diet": "string",
                "precautionary_measures": "string"
            }},
            "cleaned_input": "string"
        }}
        """
        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            response_text = response.text.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
                
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise

# Initialize predictor
try:
    predictor = DiseasePredictor()
except Exception as e:
    logger.error(f"Failed to initialize predictor: {str(e)}")
    predictor = None

class DoctorRecommender:
    def __init__(self, data_dir="data", *args, **kwargs):
        self.data_dir = Path(data_dir)

        self.doctors_df = None
        self.specialties = []

        self.load_resources()

    def load_resources(self):
        """Load data for doctor recommendation."""
        try:
            data_path = self.data_dir / "kolkata_doctors_dataset.csv"

            if not data_path.exists():
                raise FileNotFoundError("Doctor recommendation data file not found.")

            self.doctors_df = pd.read_csv(data_path)
            self.specialties = self.doctors_df["Specialty"].str.lower().unique()

            logger.info("✅ Doctor recommender resources loaded successfully")

        except Exception as e:
            logger.error(f"❌ Error loading doctor recommender resources: {str(e)}")
            raise

    def recommend(self, symptom_input):
        """Recommend doctors based on symptoms."""
        user_input = symptom_input.lower().strip()
        try:
            corrected = str(TextBlob(user_input).correct())
        except Exception:
            corrected = user_input

        if corrected in self.specialties:
            specialty = corrected
        else:
            # Predict specialty using Gemini
            specialties_list = ", ".join(self.specialties)
            prompt = f"Given the symptom '{corrected}', which medical specialty from this exact list is most appropriate? List: [{specialties_list}]. Return ONLY the specialty name, nothing else."
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            specialty = response.text.strip().lower()

        matching = self.doctors_df[self.doctors_df["Specialty"].str.lower().str.strip() == specialty.lower()]
        return matching.to_dict(orient="records")

try:
    doctor_recommender = DoctorRecommender()
except Exception as e:
    logger.error(f"Failed to initialize doctor recommender: {str(e)}")
    doctor_recommender = None

@app.route("/")
def home():
    """Home page"""
    return render_template("index.html")

@app.route("/health")
def health_check():
    """Health check endpoint"""
    status = "healthy" if predictor else "unhealthy"
    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor is not None
    })

@app.route("/model-info")
def model_info():
    """Get model information"""
    if not predictor or not predictor.model_metadata:
        return jsonify({"error": "Model metadata not available"}), 404
    
    return jsonify(predictor.model_metadata)

@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    """Predict disease from symptoms"""
    if not predictor:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            raise BadRequest("No JSON data provided")
        
        symptom_input = data.get("symptom", "").strip()
        if not symptom_input:
            raise BadRequest("Symptom input is required")
        
        # Validate input length
        if len(symptom_input) > 1000:
            raise BadRequest("Symptom input too long (max 1000 characters)")
        
        # Make prediction
        prediction_result = predictor.predict_disease(symptom_input)
        
        # Prepare response
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "input": {
                "original": symptom_input,
                "cleaned": prediction_result.get('cleaned_input', symptom_input)
            },
            "prediction": {
                "primary_disease": prediction_result.get('primary_disease', 'Unknown'),
                "confidence": float(prediction_result.get('confidence', 0.0)),
                "alternative_predictions": prediction_result.get('alternative_predictions', []),
                "confidence_level": get_confidence_level(float(prediction_result.get('confidence', 0.0)))
            },
            "medical_info": prediction_result.get('medical_info', {})
        }
        
        # Add disclaimer if confidence is low
        if response["prediction"]["confidence"] < 0.6:
            response["disclaimer"] = "Low confidence prediction. Please consult a healthcare professional."
        
        return jsonify(response)
        
    except BadRequest as e:
        return jsonify({"error": str(e), "success": False}), 400
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        error_msg = str(e)
        if "API key expired" in error_msg or "INVALID_ARGUMENT" in error_msg:
            return jsonify({
                "error": "The Gemini API key has expired or is invalid. Please update it.",
                "success": False
            }), 500
        return jsonify({
            "error": "Internal server error during prediction",
            "success": False
        }), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Predict diseases for multiple symptom inputs"""
    if not predictor:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        symptoms_list = data.get("symptoms", [])
        
        if not symptoms_list or len(symptoms_list) > 10:
            raise BadRequest("Provide 1-10 symptom inputs")
        
        results = []
        for i, symptoms in enumerate(symptoms_list):
            try:
                prediction_result = predictor.predict_disease(symptoms)
                results.append({
                    "index": i,
                    "success": True,
                    "prediction": prediction_result.get('primary_disease', 'Unknown'),
                    "confidence": float(prediction_result.get('confidence', 0.0))
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        return jsonify({
            "success": True,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except BadRequest as e:
        return jsonify({"error": str(e), "success": False}), 400
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500

def get_confidence_level(confidence):
    """Convert confidence score to human-readable level"""
    if confidence >= 0.8:
        return "High"
    elif confidence >= 0.6:
        return "Medium"
    elif confidence >= 0.4:
        return "Low"
    else:
        return "Very Low"
    

@app.route("/BMI")
def BMI_Checker():
    return render_template('bmi.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/prescription")
def prescription():
    return render_template('prescription.html')

@app.route("/Contact")
def contact():
    return render_template('contact.html')

@app.route("/Doctor")
def doctor():
    return render_template('doctor.html')


@app.route("/api/doctors", methods=["POST"])
def recommend_doctors():
    if not doctor_recommender:
        return jsonify({"error": "Doctor recommender not available"}), 503

    data = request.get_json()
    user_input = data.get("symptom", "")

    if not user_input.strip():
        return jsonify({"error": "Symptom input is required"}), 400

    recommendations = doctor_recommender.recommend(user_input)
    return jsonify(recommendations)


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    
    # Run with proper configuration
    app.run(
        debug=True,  # Set to False for production
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000))
    )