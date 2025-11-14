from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import requests

load_dotenv()

app = Flask(__name__)
CORS(app)

# Disease classes (15-class PlantVillage subset)
# NOTE: This MUST match the exact order your model was trained with!
# If predictions seem wrong, we need to verify the correct class order
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy'
]

# Initialize model variable
model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL_ENV = os.getenv('GROQ_MODEL')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODELS = [
    "llama-3.2-3b-preview",
    "llama-3.2-1b-preview",
    "llama-3.2-11b-text-preview",
    "llama-3-8b-8192",
    "llama-3-70b-8192",
    "deepseek-r1-distill-llama-70b",
    "mixtral-8x7b-32768",
]


def _groq_model_priority():
    priority = []
    if GROQ_MODEL_ENV:
        priority.append(GROQ_MODEL_ENV)
    priority.extend([m for m in DEFAULT_GROQ_MODELS if m not in priority])
    return priority


def _call_groq_chat(model_name, plant, disease, confidence):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "temperature": 0.3,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an agronomy expert. "
                    "Always respond in JSON with practical, brief advice for farmers."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "plant": plant,
                        "disease": disease,
                        "confidence": confidence,
                        "instructions": {
                            "format": "json",
                            "fields": {
                                "description": "Short summary of what this diagnosis means",
                                "severity": "one of: none, low, moderate, high, critical",
                                "remedies": "array of 3-5 numbered treatment recommendations",
                                "warnings": "optional array highlighting risks or spread",
                                "follow_up": "short guidance on what to monitor next",
                            },
                        },
                    }
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def fetch_groq_advice(disease_name, confidence):
    plant = disease_name.split('___')[0] if '___' in disease_name else disease_name
    for model_name in _groq_model_priority():
        try:
            advice = _call_groq_chat(model_name, plant, disease_name, confidence)
            advice["source"] = "groq"
            advice["model"] = model_name
            return advice
        except Exception as error:
            print(f"⚠️  Groq model {model_name} failed: {error}")
            continue

    return {
        "description": f"Detected {disease_name}.",
        "severity": "unknown",
        "remedies": [
            "Capture a clearer image with good lighting",
            "Isolate the plant from healthy specimens until diagnosis is confirmed",
            "Consult a local agronomy expert for an in-depth assessment",
        ],
        "follow_up": "Re-run the diagnosis after you gather more information.",
        "source": "fallback",
    }

def load_model():
    """Load the pretrained model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            # Load with custom options to handle TensorFlow version differences
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False  # Don't compile, we'll do it manually
            )
            # Recompile with updated parameters
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy']
            )
            print("✅ Model loaded and compiled successfully!")
            print(f"📊 Model input shape: {model.input_shape}")
            print(f"📊 Model output classes: {model.output_shape[-1]}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
            print("Using dummy predictions for development.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using dummy predictions for development.")

def preprocess_image(image_bytes):
    """Preprocess image for model input"""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((256, 256))  # Model expects 256x256
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def index():
    """Health check endpoint"""
    return "Hello, World!"
    
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict crop disease from image"""
    try:
        # Check if image is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Validate image quality
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img)
        
        # Check if image is too dark
        brightness = np.mean(img_array)
        if brightness < 30:  # Very dark image
            return jsonify({
                'success': False,
                'error': 'Image too dark',
                'message': 'Please capture image with better lighting',
                'brightness': float(brightness)
            }), 400
        
        # Check if image has enough variance (not blank)
        variance = np.var(img_array)
        if variance < 100:  # Low variance = blank/uniform image
            return jsonify({
                'success': False,
                'error': 'Image lacks detail',
                'message': 'Please capture a clear plant image',
                'variance': float(variance)
            }), 400
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        if model is not None:
            predictions = model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # CONFIDENCE THRESHOLD CHECK
            MIN_CONFIDENCE = 0.70  # 70% minimum confidence
            if confidence < MIN_CONFIDENCE:
                return jsonify({
                    'success': False,
                    'error': 'Low confidence prediction',
                    'message': 'Unable to identify plant with confidence. Please capture a clearer image of plant leaves.',
                    'confidence': confidence,
                    'suggestion': 'Ensure good lighting and focus on plant leaves'
                }), 400
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            top_predictions = []
            for idx in top_3_idx:
                class_name = DISEASE_CLASSES[idx]
                top_predictions.append({
                    'disease': class_name,
                    'confidence': float(predictions[0][idx]),
                })
        else:
            # Dummy prediction for development (when model is not available)
            predicted_class_idx = 3  # healthy apple
            confidence = 0.85
            top_predictions = [
                {
                    'disease': DISEASE_CLASSES[predicted_class_idx],
                    'confidence': confidence,
                }
            ]
        
        predicted_disease = DISEASE_CLASSES[predicted_class_idx]
        guidance = fetch_groq_advice(predicted_disease, confidence)
        plant, condition = predicted_disease.split('___')
        
        return jsonify({
            'success': True,
            'prediction': {
                'disease': predicted_disease,
                'plant': plant,
                'condition': condition,
                'confidence': confidence,
                'guidance': guidance
            },
            'top_predictions': top_predictions
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Starting Flask server...")
    # Use use_reloader=False to avoid sandbox permission issues
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

