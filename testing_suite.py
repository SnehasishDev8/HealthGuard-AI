# test_disease_predictor.py
import unittest
import json
import tempfile
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

from main_app import app, DiseasePredictor
from train_disease import DiseaseModelTrainer

class TestDiseasePredictor(unittest.TestCase):
    """Test cases for the disease prediction system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.data_dir = Path(self.temp_dir) / "data"
        self.models_dir.mkdir()
        self.data_dir.mkdir()
        
        # Create sample data for testing
        self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for testing"""
        # Sample disease data
        sample_diseases = pd.DataFrame({
            'Symptoms': [
                "['fever', 'headache', 'body ache']",
                "['cough', 'shortness of breath', 'fever']",
                "['nausea', 'vomiting', 'stomach pain']",
                "['rash', 'itching', 'swelling']",
                "['chest pain', 'shortness of breath', 'fatigue']"
            ],
            'Disease': [
                'Flu',
                'Pneumonia',
                'Gastroenteritis',
                'Allergic Reaction',
                'Heart Disease'
            ]
        })
        
        # Sample medical information
        sample_med_info = pd.DataFrame({
            'Disease': ['Flu', 'Pneumonia', 'Gastroenteritis'],
            'Treatment': ['Rest and fluids', 'Antibiotics', 'Hydration'],
            'Medicinal Composition': ['Paracetamol', 'Amoxicillin', 'ORS'],
            'Ingredients to Avoid': ['Alcohol', 'Dairy', 'Spicy food'],
            'Recommended Diet': ['Light foods', 'Protein rich', 'BRAT diet'],
            'Precautionary Measures': ['Rest', 'Avoid crowds', 'Hygiene']
        })
        
        # Save sample data
        sample_diseases.to_csv(self.data_dir / "decease.csv", index=False)
        sample_med_info.to_csv(self.data_dir / "med.csv", index=False)
    
    def test_model_training(self):
        """Test the model training process"""
        trainer = DiseaseModelTrainer(
            data_path=str(self.data_dir / "decease.csv"),
            models_dir=str(self.models_dir)
        )
        
        # Test data loading
        self.assertTrue(trainer.load_and_validate_data())
        self.assertIsNotNone(trainer.df)
        self.assertEqual(len(trainer.df), 5)
        
        # Test preprocessing
        trainer.preprocess_data()
        self.assertTrue(all(trainer.df["cleaned_symptoms"].str.len() > 0))
        
        # Test model training
        accuracy = trainer.train_model()
        self.assertGreater(accuracy, 0.0)
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.vectorizer)
        
        # Test model saving
        self.assertTrue(trainer.save_models())
        self.assertTrue((self.models_dir / "disease_model.pkl").exists())
        self.assertTrue((self.models_dir / "disease_vectorizer.pkl").exists())
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        trainer = DiseaseModelTrainer()
        
        # Test various input formats
        test_cases = [
            ("['fever', 'headache']", "fever headache"),
            ("cough, cold, sneezing", "cough cold sneezing"),
            ("FEVER AND CHILLS", "fever and chills"),
            ("", ""),
            (None, "")
        ]
        
        for input_text, expected in test_cases:
            result = trainer.clean_symptom_text(input_text)
            if expected:
                self.assertIn(expected.split()[0], result.lower())
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        # This test assumes we have trained models
        try:
            predictor = DiseasePredictor(
                models_dir=str(self.models_dir),
                data_dir=str(self.data_dir)
            )
            # If no models exist, this should raise an exception
        except FileNotFoundError:
            self.skipTest("No trained models available for testing")
    
    def test_api_endpoints(self):
        """Test Flask API endpoints"""
        # Test health endpoint
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('timestamp', data)
        
        # Test model info endpoint
        response = self.app.get('/model-info')
        # This might return 404 if no metadata is available
        self.assertIn(response.status_code, [200, 404])
    
    def test_prediction_endpoint_validation(self):
        """Test prediction endpoint input validation"""
        # Test empty request
        response = self.app.post('/predict_disease')
        self.assertEqual(response.status_code, 400)
        
        # Test empty symptom
        response = self.app.post('/predict_disease',
                                data=json.dumps({"symptom": ""}),
                                content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Test too long input
        long_input = "symptom " * 200  # Over 1000 characters
        response = self.app.post('/predict_disease',
                                data=json.dumps({"symptom": long_input}),
                                content_type='application/json')
        self.assertEqual(response.status_code, 400)
    
    def test_batch_prediction_validation(self):
        """Test batch prediction endpoint validation"""
        # Test empty request
        response = self.app.post('/predict_batch')
        self.assertEqual(response.status_code, 400)
        
        # Test too many symptoms
        too_many_symptoms = ["fever"] * 15
        response = self.app.post('/predict_batch',
                                data=json.dumps({"symptoms": too_many_symptoms}),
                                content_type='application/json')
        self.assertEqual(response.status_code, 400)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestDiseaseModelTrainer(unittest.TestCase):
    """Test cases specifically for the model trainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / "test_data.csv"
        self.models_dir = Path(self.temp_dir) / "models"
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Symptoms': [
                "['fever', 'headache']",
                "['cough', 'cold']",
                "['nausea', 'vomiting']"
            ],
            'Disease': ['Flu', 'Cold', 'Gastro']
        })
        sample_data.to_csv(self.data_path, index=False)
    
    def test_data_validation(self):
        """Test data validation functionality"""
        trainer = DiseaseModelTrainer(
            data_path=str(self.data_path),
            models_dir=str(self.models_dir)
        )
        
        # Test successful loading
        self.assertTrue(trainer.load_and_validate_data())
        
        # Test with missing file
        trainer.data_path = "nonexistent.csv"
        self.assertFalse(trainer.load_and_validate_data())
    
    def test_symptom_cleaning(self):
        """Test symptom text cleaning"""
        trainer = DiseaseModelTrainer()
        
        test_cases = [
            ("['fever', 'headache', 'body ache']", "fever headache body ache"),
            ("cough; cold, sneezing", "cough cold sneezing"),
            ("  FEVER AND  CHILLS  ", "fever and chills"),
            ("nausea,vomiting", "nausea vomiting")
        ]
        
        for input_text, expected_words in test_cases:
            cleaned = trainer.clean_symptom_text(input_text)
            for word in expected_words.split():
                self.assertIn(word, cleaned.lower())
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.data_dir = Path(self.temp_dir) / "data"
        self.models_dir.mkdir()
        self.data_dir.mkdir()
        
        # Create comprehensive test data
        self.create_comprehensive_test_data()
        
        # Train a model for integration testing
        self.train_test_model()
    
    def create_comprehensive_test_data(self):
        """Create comprehensive test data"""
        diseases_data = pd.DataFrame({
            'Symptoms': [
                "['fever', 'headache', 'muscle pain', 'fatigue']",
                "['cough', 'shortness of breath', 'chest pain', 'fever']",
                "['nausea', 'vomiting', 'diarrhea', 'stomach cramps']",
                "['skin rash', 'itching', 'swelling', 'redness']",
                "['joint pain', 'stiffness', 'swelling', 'warmth']",
                "['chest pain', 'irregular heartbeat', 'shortness of breath']",
                "['frequent urination', 'excessive thirst', 'fatigue']",
                "['persistent cough', 'weight loss', 'night sweats', 'fever']"
            ],
            'Disease': [
                'Influenza',
                'Pneumonia',
                'Gastroenteritis',
                'Allergic Reaction',
                'Arthritis',
                'Heart Disease',
                'Diabetes',
                'Tuberculosis'
            ]
        })
        
        medical_info = pd.DataFrame({
            'Disease': ['Influenza', 'Pneumonia', 'Gastroenteritis', 'Allergic Reaction'],
            'Treatment': [
                'Rest, fluids, antiviral medication',
                'Antibiotics, oxygen therapy',
                'Hydration, electrolyte replacement',
                'Antihistamines, avoid allergens'
            ],
            'Medicinal Composition': [
                'Oseltamivir, Paracetamol',
                'Amoxicillin, Oxygen',
                'ORS, Probiotics',
                'Cetirizine, Epinephrine'
            ],
            'Ingredients to Avoid': [
                'Alcohol, Aspirin in children',
                'Alcohol, Smoking',
                'Dairy, Spicy foods',
                'Known allergens'
            ],
            'Recommended Diet': [
                'Light, easily digestible foods',
                'High protein, vitamin C',
                'BRAT diet, clear liquids',
                'Avoid trigger foods'
            ],
            'Precautionary Measures': [
                'Rest, isolation, hygiene',
                'Rest, avoid smoking',
                'Hand hygiene, safe food',
                'Carry epinephrine, avoid triggers'
            ]
        })
        
        diseases_data.to_csv(self.data_dir / "decease.csv", index=False)
        medical_info.to_csv(self.data_dir / "med.csv", index=False)
    
    def train_test_model(self):
        """Train a model for integration testing"""
        trainer = DiseaseModelTrainer(
            data_path=str(self.data_dir / "decease.csv"),
            models_dir=str(self.models_dir)
        )
        
        trainer.load_and_validate_data()
        trainer.preprocess_data()
        trainer.train_model()
        trainer.save_models()
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction workflow"""
        # Initialize predictor with trained model
        predictor = DiseasePredictor(
            models_dir=str(self.models_dir),
            data_dir=str(self.data_dir)
        )
        
        # Test prediction
        test_symptoms = "I have fever and headache and muscle pain"
        result = predictor.predict_disease(test_symptoms)
        
        # Verify prediction structure
        self.assertIn('primary_prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('all_predictions', result)
        self.assertIsInstance(result['confidence'], float)
        self.assertGreater(result['confidence'], 0)
        
        # Test disease info retrieval
        disease_info = predictor.get_disease_info(result['primary_prediction'])
        self.assertIsInstance(disease_info, dict)
    
    def test_api_with_trained_model(self):
        """Test API endpoints with a trained model"""
        # Mock the global predictor with our test predictor
        import main_app as main
        original_predictor = main.predictor
        
        try:
            main.predictor = DiseasePredictor(
                models_dir=str(self.models_dir),
                data_dir=str(self.data_dir)
            )
            
            app = main.app.test_client()
            app.testing = True
            
            # Test successful prediction
            response = app.post('/predict_disease',
                               data=json.dumps({"symptom": "fever and headache"}),
                               content_type='application/json')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertTrue(data['success'])
            self.assertIn('prediction', data)
            self.assertIn('primary_disease', data['prediction'])
            
        finally:
            # Restore original predictor
            main.predictor = original_predictor
    
    def tearDown(self):
        """Clean up integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDiseasePredictor,
        TestDiseaseModelTrainer,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)