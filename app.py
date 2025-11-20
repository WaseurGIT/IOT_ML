from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import io
import os
import json
import torch
from torchvision import models, transforms
import numpy as np
import requests
import threading
from huggingface_hub import hf_hub_download, login

load_dotenv()

# HuggingFace Authentication
HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("✅ Authenticated with HuggingFace")
    except Exception as e:
        print(f"⚠️  HuggingFace authentication failed: {e}")
else:
    print("⚠️  No HuggingFace token found. Set HF_TOKEN or HUGGINGFACE_TOKEN environment variable.")
    print("   For gated repositories, you need to:")
    print("   1. Create a HuggingFace account at https://huggingface.co")
    print("   2. Request access to the model repository")
    print("   3. Generate a token at https://huggingface.co/settings/tokens")
    print("   4. Set it as: export HF_TOKEN='your_token_here'")

app = Flask(__name__)
CORS(app)

# Global variables for model, transform, and class mapping
model = None  # BD Crop Disease Model
transform = None  # Image preprocessing transform
class_names = {}  # Class mapping (94 classes)
_models_loading = False  # Flag to prevent concurrent loading
_models_lock = threading.Lock()  # Lock for thread-safe model loading

# Load models lazily on first request (for gunicorn/production)
# This prevents blocking during worker startup
def ensure_models_loaded():
    """Ensure model is loaded (lazy loading for production) - thread-safe"""
    global _models_loading
    
    # Quick check without lock (most common case)
    if model is not None:
        return
    
    # Acquire lock to prevent concurrent loading
    with _models_lock:
        # Double-check after acquiring lock
        if model is not None:
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
            print("🔄 Model not loaded yet, loading now...")
            load_models()
            if model is None:
                print("⚠️  Warning: Model not loaded after load_models() call")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            # Don't crash - service can still respond (will return error on predict)
        finally:
            _models_loading = False

# Automatically kick off background loading (prevents Render 502 during probes)
AUTO_LOAD_MODELS = os.getenv('AUTO_LOAD_MODELS', 'true').strip().lower() in (
    '1', 'true', 'yes', 'on'
)


def _background_model_loader():
    try:
        ensure_models_loaded()
        print("✅ Background model loading completed")
    except Exception as e:
        print(f"❌ Background model loading failed: {e}")


if AUTO_LOAD_MODELS:
    threading.Thread(target=_background_model_loader, daemon=True).start()
    print("⚙️ Background model loading thread started")

# BD Crop Disease Model Configuration
MODEL_REPO_ID = "Saon110/bd-crop-vegetable-plant-disease-model"
MODEL_FILENAME = "crop_veg_plant_disease_model.pth"
CLASS_MAPPING_FILENAME = "class_mapping.json"

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

def load_class_mapping():
    """Load class mapping from HuggingFace or use fallback"""
    global class_names
    
    try:
        # Try to download from HuggingFace
        token = HF_TOKEN if HF_TOKEN else None
        class_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=CLASS_MAPPING_FILENAME,
            token=token
        )
        with open(class_path, 'r') as f:
            class_names = json.load(f)
        print(f"✅ Loaded class mapping from HuggingFace ({len(class_names)} classes)")
    except Exception as e:
        print(f"⚠️  Could not load class mapping from HuggingFace: {e}")
        print("📋 Using fallback class mapping...")
        # Fallback class mapping (94 classes)
        class_names = {
            "0": "Banana_Black_Pitting_or_Banana_Rust",
            "1": "Banana_Crown_Rot",
            "2": "Banana_Healthy",
            "3": "Banana_fungal_disease",
            "4": "Banana_leaf_Banana_Scab_Moth",
            "5": "Banana_leaf_Black_Sigatoka",
            "6": "Banana_leaf_Healthy",
            "7": "Banana_leaf__Black_Leaf_Streak",
            "8": "Banana_leaf__Panama_Disease.",
            "9": "Cauliflower_Bacterial_spot_rot",
            "10": "Cauliflower_Black_Rot",
            "11": "Cauliflower_Downy_Mildew",
            "12": "Cauliflower_Healthy",
            "13": "Corn_Blight",
            "14": "Corn_Common_Rust",
            "15": "Corn_Gray_Leaf_Spot",
            "16": "Corn_Healthy",
            "17": "Cotton_Aphids",
            "18": "Cotton_Army worm",
            "19": "Cotton_Bacterial blight",
            "20": "Cotton_Healthy",
            "21": "Guava_fruit_Anthracnose",
            "22": "Guava_fruit_Healthy",
            "23": "Guava_fruit_Scab",
            "24": "Guava_fruit_Styler_end_root",
            "25": "Guava_leaf_Anthracnose",
            "26": "Guava_leaf_Canker",
            "27": "Guava_leaf_Dot",
            "28": "Guava_leaf_Healthy",
            "29": "Guava_leaf_Rust",
            "30": "Jute_Cescospora Leaf Spot",
            "31": "Jute_Golden Mosaic",
            "32": "Jute_Healthy Leaf",
            "33": "Mango_Anthracnose",
            "34": "Mango_Bacterial_Canker",
            "35": "Mango_Cutting_Weevil",
            "36": "Mango_Gall_Midge",
            "37": "Mango_Healthy",
            "38": "Mango_Powdery_Mildew",
            "39": "Mango_Sooty_Mould",
            "40": "Mango_die_back",
            "41": "Papaya_Anthracnose",
            "42": "Papaya_BacterialSpot",
            "43": "Papaya_Curl",
            "44": "Papaya_Healthy",
            "45": "Papaya_Mealybug",
            "46": "Papaya_Mite_disease",
            "47": "Papaya_Mosaic",
            "48": "Papaya_Ringspot",
            "49": "Potato_Black_Scurf",
            "50": "Potato_Blackleg",
            "51": "Potato_Blackspot_Bruising",
            "52": "Potato_Brown_Rot",
            "53": "Potato_Common_Scab",
            "54": "Potato_Dry_Rot",
            "55": "Potato_Healthy_Potatoes",
            "56": "Potato_Miscellaneous",
            "57": "Potato_Pink_Rot",
            "58": "Potato_Soft_Rot",
            "59": "Rice_Blast",
            "60": "Rice_Brownspot",
            "61": "Rice_Tungro",
            "62": "Rice_bacterial_leaf_blight",
            "63": "Rice_bacterial_leaf_streak",
            "64": "Rice_bacterial_panicle_blight",
            "65": "Rice_dead_heart",
            "66": "Rice_downy_mildew",
            "67": "Rice_hispa",
            "68": "Rice_normal",
            "69": "Sugarcane_Healthy",
            "70": "Sugarcane_Mosaic",
            "71": "Sugarcane_RedRot",
            "72": "Sugarcane_Rust",
            "73": "Sugarcane_Yellow",
            "74": "Tea_Anthracnose",
            "75": "Tea_algal_leaf",
            "76": "Tea_bird_eye_spot",
            "77": "Tea_brown_blight",
            "78": "Tea_gray_light",
            "79": "Tea_healthy",
            "80": "Tea_red_leaf_spot",
            "81": "Tea_white_spot",
            "82": "Tomato_Bacterial_Spot",
            "83": "Tomato_Early_Blight",
            "84": "Tomato_Late_Blight",
            "85": "Tomato_Leaf_Mold",
            "86": "Tomato_Septoria_Leaf_Spot",
            "87": "Tomato_Spider_Mites_Two-spotted_Spider_Mite",
            "88": "Tomato_Target_Spot",
            "89": "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
            "90": "Tomato_healthy",
            "91": "Wheat_Healthy",
            "92": "Wheat_septoria",
            "93": "Wheat_stripe_rust"
        }
        print(f"✅ Using fallback class mapping ({len(class_names)} classes)")

def load_models():
    """Load BD Crop Disease Model"""
    global model, transform, class_names
    
    print(f"\n{'='*60}")
    print(f"🎯 Loading BD Crop & Vegetable Plant Disease Model")
    print(f"📦 Model: {MODEL_REPO_ID}")
    print(f"{'='*60}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 1. Download model weights
        print("📥 Downloading model weights...")
        # Use token if available
        token = HF_TOKEN if HF_TOKEN else None
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            token=token
        )
        print(f"✅ Model weights downloaded: {model_path}")
        
        # 2. Setup model architecture (matching training exactly)
        print("🏗️  Setting up model architecture...")
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 94)
        )
        
        # 3. Load model weights with robust handling
        print("📂 Loading model weights...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Handle DataParallel wrapper (remove 'module.' prefix if present)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load the state dictionary
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("✅ Model weights loaded successfully")
        
        # 4. Define preprocessing transforms (same as validation)
        print("🔄 Setting up image preprocessing...")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("✅ Preprocessing transforms ready")
        
        # 5. Load class mapping
        load_class_mapping()
        
        print(f"\n{'='*60}")
        print(f"✅ BD Crop Disease Model loaded successfully")
        print(f"💾 Device: {device}")
        print(f"📊 Classes: {len(class_names)}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        transform = None

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
    ensure_models_loaded()  # Lazy load model on first request
    return jsonify({
        'status': 'running',
        'service': 'BD Crop & Vegetable Plant Disease Detection ML Service',
        'model': 'BD Crop Disease Model (ResNet50)',
        'model_loaded': model is not None,
        'num_classes': len(class_names),
        'version': '4.0'
    })

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    ensure_models_loaded()  # Lazy load model on first request
    return jsonify({
        'status': 'healthy' if model is not None else 'loading',
        'model': 'BD Crop Disease Model (ResNet50)',
        'model_loaded': model is not None,
        'num_classes': len(class_names),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_repo': MODEL_REPO_ID
    })

def predict_image(img):
    """Run prediction on BD Crop Disease Model"""
    global model, transform, class_names
    
    if model is None or transform is None:
        return None
    
    # Preprocess image
    image_tensor = transform(img).unsqueeze(0)
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Get top prediction
    confidence, predicted_idx = torch.max(probabilities, dim=0)
    predicted_idx = predicted_idx.item()
    confidence = confidence.item()
    
    # Get class name
    disease_name = class_names.get(str(predicted_idx), f"Unknown_Class_{predicted_idx}")
    
    return {
        'disease': disease_name,
        'confidence': confidence,
        'model': 'bd_crop_disease_model'
    }

@app.route('/predict', methods=['POST'])
def predict():
    """Predict crop disease from image using BD Crop Disease Model"""
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
        
        # Ensure model is loaded
        print("🔄 Ensuring model is loaded...")
        try:
            ensure_models_loaded()
        except Exception as e:
            print(f"❌ Error ensuring model loaded: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': 'Failed to load model',
                'details': str(e)
            }), 500
        
        # Check if model loaded
        if model is None:
            print("⚠️  Model not loaded after ensure_models_loaded()")
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please check service logs.'
            }), 500
        
        print(f"✅ Model ready: BD Crop Disease Model")
        
        # Run prediction
        result = predict_image(img)
        
        if not result:
            return jsonify({'success': False, 'error': 'Prediction failed'}), 500
        
        disease_name = result['disease']
        confidence = result['confidence']
        model_used = result['model']
        
        print(f"🔍 Final Prediction: {disease_name} ({confidence*100:.1f}%)")
        
        # Check confidence threshold (configurable via env, default 0.5)
        CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
        low_confidence = confidence < CONFIDENCE_THRESHOLD
        
        if low_confidence:
            print(f"⚠️  Low confidence: {confidence:.2f} < {CONFIDENCE_THRESHOLD} (threshold)")
            print(f"   Still returning prediction with warning flag")
        
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
                'low_confidence': low_confidence,
            }
        }
        
        # Add warning message if low confidence
        if low_confidence:
            response['warning'] = f'Low confidence prediction ({confidence*100:.1f}%). Consider retaking the image for better accuracy.'
            response['suggestion'] = 'Please provide a clearer, well-lit image with better focus on the plant leaf for more accurate results.'
        
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
    print("🌾 BD Crop & Vegetable Plant Disease Detection ML Service")
    print("="*70)
    
    # Load model at startup
    load_models()
    
    print("\nStarting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

