from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification
from dotenv import load_dotenv
from PIL import Image
import io
import os
import json
import torch
import numpy as np
import requests
import threading

load_dotenv()

app = Flask(__name__)
CORS(app)

# Global variables for models and processors
models = {}  # Dict to store multiple models
processors = {}  # Dict to store multiple processors
_models_loading = False  # Flag to prevent concurrent loading
_models_lock = threading.Lock()  # Lock for thread-safe model loading

# Load models lazily on first request (for gunicorn/production)
# This prevents blocking during worker startup
def ensure_models_loaded():
    """Ensure models are loaded (lazy loading for production) - thread-safe"""
    global _models_loading
    
    # Quick check without lock (most common case)
    if models:
        return
    
    # Acquire lock to prevent concurrent loading
    with _models_lock:
        # Double-check after acquiring lock
        if models:
            return
        
        # Check if another thread is already loading
        if _models_loading:
            # Wait for other thread to finish loading
            while _models_loading:
                threading.Event().wait(0.1)
            return
        
        # Mark as loading
        _models_loading = True
        try:
            print("🔄 Models not loaded yet, loading now...")
            load_models()
            if not models:
                print("⚠️  Warning: No models loaded after load_models() call")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            # Don't crash - service can still respond (will return error on predict)
        finally:
            _models_loading = False

# ============================================================================
# 🎯 AVAILABLE MODELS - Choose the best for your needs:
# ============================================================================
AVAILABLE_MODELS = {
    # ✅ Option 1: WORKING - ViT Base (13 classes, ~350MB, CPU-friendly)
    'vit_base': "wambugu1738/crop_leaf_diseases_vit",
    
    # ✅ Option 2: WORKING - ResNet50 (38 classes, ~100MB, accurate)
    'resnet50': "rajistics/finetuned-plant-disease-identification",
    
    # ✅ Option 3: WORKING - MobileNet (38 classes, ~15MB, very fast)
    'mobilenet': "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    
    # 🔬 Option 4: EXPERIMENTAL - DenseNet (for testing)
    'densenet': "nateraw/vit-base-beans",
}

# ============================================================================
# 🎮 MULTI-MODEL CONFIGURATION:
# ============================================================================
# Choose your mode:
MULTI_MODEL_MODE = 'single'  # Options: 'single', 'ensemble', 'all'

# SINGLE MODE: Load only one model (fast, low memory)
SINGLE_MODEL = 'vit_base'  # Which model to use in single mode

# ENSEMBLE MODE: Load multiple models and combine predictions (most accurate)
ENSEMBLE_MODELS = ['vit_base', 'resnet50']  # Models to use in ensemble

# ALL MODE: Load all models, let user choose per request
# ⚠️ WARNING: Uses 500MB+ RAM, slower startup

# 📊 MODE COMPARISON:
# - single:   Fast, 350MB RAM, 100-300ms ✅ CURRENT
# - ensemble: Most accurate, 450MB RAM, 300-800ms (votes from 2+ models)
# - all:      Flexible, 500MB+ RAM, varies (user picks per request)

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

    prompt_payload = {
        "model": model_name,
        "temperature": 0.3,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an agronomy assistant helping farmers diagnose and treat crop diseases. "
                    "Respond with concise, actionable advice. Always provide JSON."
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
                                "description": "Short explanation of the disease or health status",
                                "severity": "one of: none, low, moderate, high, critical",
                                "remedies": "array of 3-5 practical treatment steps",
                                "warnings": "optional array of things to watch out for",
                                "follow_up": "short suggestion for continued care",
                            },
                        },
                    }
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=prompt_payload, timeout=15)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def fetch_groq_advice(disease_name, confidence):
    plant = disease_name.split("___")[0] if "___" in disease_name else disease_name
    for model_name in _groq_model_priority():
        try:
            groq_response = _call_groq_chat(model_name, plant, disease_name, confidence)
            groq_response["source"] = "groq"
            groq_response["model"] = model_name
            return groq_response
        except Exception as groq_error:
            print(f"⚠️  Groq model {model_name} failed: {groq_error}")
            continue

    return {
        "description": f"Detected {disease_name}.",
        "severity": "unknown",
        "remedies": [
            "Retake a clear, well-lit photo of the plant leaf",
            "Consult a local agronomist or agricultural extension office",
            "Monitor the plant over the next 48 hours for symptom progression",
        ],
        "follow_up": "Re-run diagnosis once you gather more information.",
        "source": "fallback",
    }

def load_models():
    """Load model(s) based on MULTI_MODEL_MODE"""
    global models, processors
    
    # Determine which models to load
    if MULTI_MODEL_MODE == 'single':
        models_to_load = [SINGLE_MODEL]
    elif MULTI_MODEL_MODE == 'ensemble':
        models_to_load = ENSEMBLE_MODELS
    elif MULTI_MODEL_MODE == 'all':
        models_to_load = list(AVAILABLE_MODELS.keys())
    else:
        print(f"❌ Invalid MULTI_MODEL_MODE: {MULTI_MODEL_MODE}")
        return
    
    print(f"\n{'='*60}")
    print(f"🎯 Loading models in '{MULTI_MODEL_MODE}' mode")
    print(f"📦 Models to load: {models_to_load}")
    print(f"{'='*60}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load each model
    for model_key in models_to_load:
        if model_key not in AVAILABLE_MODELS:
            print(f"⚠️  Skipping unknown model: {model_key}")
            continue
            
        model_name = AVAILABLE_MODELS[model_key]
        try:
            print(f"Loading {model_key}: {model_name}...")
            
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)
            model.to(device)
            model.eval()
            
            # Store in dictionaries
            models[model_key] = model
            processors[model_key] = processor
            
            print(f"✅ {model_key} loaded ({len(model.config.id2label)} classes)")
            
        except Exception as e:
            print(f"❌ Failed to load {model_key}: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next model instead of crashing
    
    print(f"\n{'='*60}")
    print(f"✅ Loaded {len(models)}/{len(models_to_load)} models successfully")
    print(f"💾 Device: {device}")
    print(f"{'='*60}\n")

def validate_image(img):
    """
    Validate image quality before prediction
    Returns: (is_valid, reason)
    """
    # Convert to numpy array
    img_array = np.array(img)
    
    # 1. Check if image is too dark
    brightness = np.mean(img_array)
    if brightness < 30:
        return False, f"Image too dark (brightness: {brightness:.1f}/255)"
    
    # 2. Check if image has enough detail (variance)
    variance = np.var(img_array)
    if variance < 100:
        return False, f"Image lacks detail (variance: {variance:.1f})"
    
    # 3. Check image size
    if img.size[0] < 50 or img.size[1] < 50:
        return False, f"Image too small ({img.size[0]}x{img.size[1]})"
    
    return True, "OK"

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    ensure_models_loaded()  # Lazy load models on first request
    return jsonify({
        'status': 'running',
        'service': 'Crop Disease Detection ML Service (Multi-Model)',
        'mode': MULTI_MODEL_MODE,
        'models_loaded': list(models.keys()),
        'version': '3.0'
    })

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    ensure_models_loaded()  # Lazy load models on first request
    return jsonify({
        'status': 'healthy',
        'mode': MULTI_MODEL_MODE,
        'models_loaded': list(models.keys()),
        'num_models': len(models),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'available_models': list(AVAILABLE_MODELS.keys())
    })

def predict_single_model(img, model_key):
    """Run prediction on a single model"""
    if model_key not in models or model_key not in processors:
        return None
    
    model = models[model_key]
    processor = processors[model_key]
    
    # Preprocess
    inputs = processor(images=img, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_idx = torch.max(probabilities, dim=-1)
    
    disease_name = model.config.id2label[predicted_idx.item()]
    confidence = confidence.item()
    
    return {
        'disease': disease_name,
        'confidence': confidence,
        'model': model_key
    }

def ensemble_predict(img, model_keys):
    """Combine predictions from multiple models using voting"""
    predictions = []
    
    print(f"🗳️  Running ensemble prediction with {len(model_keys)} models")
    
    for model_key in model_keys:
        result = predict_single_model(img, model_key)
        if result:
            predictions.append(result)
            print(f"  {model_key}: {result['disease']} ({result['confidence']*100:.1f}%)")
    
    if not predictions:
        return None
    
    # Voting: count disease predictions
    from collections import Counter
    disease_votes = Counter([p['disease'] for p in predictions])
    winning_disease = disease_votes.most_common(1)[0][0]
    
    # Average confidence for winning disease
    winning_predictions = [p for p in predictions if p['disease'] == winning_disease]
    avg_confidence = sum(p['confidence'] for p in winning_predictions) / len(winning_predictions)
    
    print(f"🏆 Ensemble result: {winning_disease} ({avg_confidence*100:.1f}%)")
    
    return {
        'disease': winning_disease,
        'confidence': avg_confidence,
        'votes': dict(disease_votes),
        'individual_predictions': predictions
    }

@app.route('/predict', methods=['POST'])
def predict():
    """Predict crop disease from image (multi-model support)"""
    try:
        print("📥 Received prediction request")
        
        # Check if image is in request
        if 'image' not in request.files:
            print("❌ No image in request")
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Read image
        try:
            image_file = request.files['image']
            image_bytes = image_file.read()
            
            if not image_bytes:
                print("❌ Empty image file")
                return jsonify({
                    'success': False,
                    'error': 'Empty image file'
                }), 400
            
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            print(f"📸 Received image: {img.size}, mode: {img.mode}, size: {len(image_bytes)} bytes")
        except Exception as e:
            print(f"❌ Error reading image: {e}")
            return jsonify({
                'success': False,
                'error': f'Invalid image file: {str(e)}'
            }), 400
        
        # Validate image quality
        is_valid, reason = validate_image(img)
        if not is_valid:
            print(f"⚠️  Image validation failed: {reason}")
            return jsonify({
                'success': False,
                'error': f'Invalid image: {reason}',
                'suggestion': 'Please provide a clear, well-lit image of a plant leaf'
            }), 400
        
        # Ensure models are loaded
        print("🔄 Ensuring models are loaded...")
        try:
            ensure_models_loaded()
        except Exception as e:
            print(f"❌ Error ensuring models loaded: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': 'Failed to load models',
                'details': str(e)
            }), 500
        
        # Check if any models loaded
        if not models:
            print("⚠️  No models loaded after ensure_models_loaded()")
            return jsonify({
                'success': False,
                'error': 'No models loaded. Please check service logs.'
            }), 500
        
        print(f"✅ Models ready: {list(models.keys())}")
        
        # Check if user specified a model (for 'all' mode)
        requested_model = request.form.get('model', None)
        
        # Determine prediction strategy
        if MULTI_MODEL_MODE == 'single' or requested_model:
            # Single model prediction
            model_key = requested_model if requested_model and requested_model in models else SINGLE_MODEL
            result = predict_single_model(img, model_key)
            
            if not result:
                return jsonify({'success': False, 'error': 'Prediction failed'}), 500
            
            disease_name = result['disease']
            confidence = result['confidence']
            model_used = result['model']
            individual_predictions = None
            
        elif MULTI_MODEL_MODE == 'ensemble':
            # Ensemble prediction
            result = ensemble_predict(img, list(models.keys()))
            
            if not result:
                return jsonify({'success': False, 'error': 'Ensemble prediction failed'}), 500
            
            disease_name = result['disease']
            confidence = result['confidence']
            model_used = 'ensemble'
            individual_predictions = result.get('individual_predictions')
            
        elif MULTI_MODEL_MODE == 'all':
            # Default to first loaded model if no model specified
            model_key = list(models.keys())[0]
            result = predict_single_model(img, model_key)
            
            if not result:
                return jsonify({'success': False, 'error': 'Prediction failed'}), 500
            
            disease_name = result['disease']
            confidence = result['confidence']
            model_used = result['model']
            individual_predictions = None
        
        print(f"🔍 Final Prediction: {disease_name} ({confidence*100:.1f}%)")
        
        # Check confidence threshold
        CONFIDENCE_THRESHOLD = 0.70
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"⚠️  Low confidence: {confidence:.2f} < {CONFIDENCE_THRESHOLD}")
            return jsonify({
                'success': False,
                'error': f'Low confidence prediction ({confidence*100:.1f}%)',
                'suggestion': 'Please provide a clearer image with better focus on the plant leaf',
                'raw_prediction': disease_name,
                'confidence': float(confidence)
            }), 400
        
        # Fetch dynamic guidance from Groq (with timeout protection)
        try:
            advice = fetch_groq_advice(disease_name, confidence)
        except Exception as e:
            print(f"⚠️  Groq API error (using fallback): {e}")
            # Use fallback advice if Groq fails
            advice = {
                "description": f"Detected {disease_name} with {confidence*100:.1f}% confidence.",
                "severity": "moderate",
                "remedies": [
                    "Monitor the plant closely",
                    "Consult a local agricultural expert",
                    "Take preventive measures based on the diagnosis"
                ],
                "follow_up": "Continue monitoring the plant's condition.",
                "source": "fallback"
            }
        
        # Build response
        response = {
            'success': True,
            'prediction': {
                'disease': disease_name,
                'confidence': float(confidence),
                'model_used': model_used,
                'guidance': advice,
            }
        }
        
        # Add individual predictions for ensemble mode
        if individual_predictions:
            response['prediction']['individual_predictions'] = individual_predictions
        
        print(f"✅ Returning prediction: {disease_name}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a more helpful error message
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            return jsonify({
                'success': False,
                'error': 'Prediction timed out. The model may be loading. Please try again in a few seconds.',
                'suggestion': 'Wait 10-15 seconds and retry the request'
            }), 504  # Gateway Timeout
        elif "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
            return jsonify({
                'success': False,
                'error': 'Service out of memory. Please try again later.',
                'suggestion': 'The service may need to restart. Wait 30 seconds and retry.'
            }), 503  # Service Unavailable
        else:
            return jsonify({
                'success': False,
                'error': f'Prediction failed: {error_msg}',
                'suggestion': 'Please check the image format and try again'
            }), 500

if __name__ == '__main__':
    print("="*70)
    print("🌱 Crop Disease Detection ML Service (Multi-Model Support)")
    print("="*70)
    
    # Load models at startup
    load_models()
    
    print("\nStarting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

