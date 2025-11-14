# ML Service for Crop Disease Prediction

This is a Flask-based microservice that provides plant disease prediction using deep learning.

**Primary App**: `app_huggingface.py` (Recommended for deployment)
- ✅ Automatically downloads models from HuggingFace
- ✅ No need for local model files
- ✅ Multiple model options (ViT, ResNet50, MobileNet)
- ✅ Supports ensemble predictions
- ✅ Better for cloud deployment

**Alternative**: `app.py` (TensorFlow/Keras)
- Requires local `model.h5` file
- Simpler but needs manual model management

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages for HuggingFace version (recommended)
pip install -r requirements_huggingface.txt
```

### 2. Start Service

```bash
# Using HuggingFace version (recommended)
python app_huggingface.py
```

Service runs on `http://localhost:5000`

Models will automatically download from HuggingFace on first run.

### 4. Test Service

```bash
# Test health
curl http://localhost:5000/health

# Test prediction (replace with your image path)
curl -X POST -F "image=@/path/to/image.jpg" http://localhost:5000/predict
```

## API Endpoints

### GET /health
Check service status

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /predict
Predict disease from image

**Request:**
- Content-Type: multipart/form-data
- Body: image file (JPEG/PNG)

**Response:**
```json
{
  "success": true,
  "prediction": {
    "disease": "Tomato___Early_blight",
    "plant": "Tomato",
    "condition": "Early_blight",
    "confidence": 0.95,
    "description": "Fungal disease...",
    "remedies": ["Remove infected leaves...", "..."],
    "severity": "moderate"
  },
  "top_predictions": [...]
}
```

## Files

- `app_huggingface.py` - **Main Flask application** (HuggingFace, recommended)
- `app.py` - Alternative TensorFlow/Keras implementation (requires model.h5)
- `requirements_huggingface.txt` - Python dependencies for HuggingFace version
- `requirements.txt` - Python dependencies for TensorFlow version

## Supported Diseases (38 classes)

### Plants
- Apple (4 classes)
- Corn/Maize (4 classes)
- Grape (4 classes)
- Tomato (10 classes)
- Potato (3 classes)
- Peach, Pepper, Cherry, Strawberry, etc.

See `app.py` for complete list.

## Model Details

### HuggingFace Version (app_huggingface.py)
- **Available Models**:
  - `vit_base`: ViT Base (13 classes, ~350MB, CPU-friendly) ✅ Default
  - `resnet50`: ResNet50 (38 classes, ~100MB, accurate)
  - `mobilenet`: MobileNet (38 classes, ~15MB, very fast)
- **Framework**: PyTorch + HuggingFace Transformers
- **Mode**: Single model (configurable to ensemble or all)

### TensorFlow Version (app.py)
- **Architecture**: MobileNetV2 (transfer learning)
- **Input Size**: 256x256 RGB
- **Output**: 15 classes (softmax)
- **Framework**: TensorFlow/Keras 2.15+

## Development

### Debug Mode
```bash
# Using HuggingFace version (recommended)
python app_huggingface.py

# Or TensorFlow version
python app.py
```

The app runs in debug mode by default on `http://localhost:5000`

## Production Deployment

### Local Production with Gunicorn
```bash
# Using HuggingFace version (recommended)
gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 2 --timeout 180 app_huggingface:app

# Or TensorFlow version
gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 2 --timeout 120 app:app
```

### Deploy to Render

#### Option 1: Using render.yaml (Recommended)

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Connect to Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml` and deploy

3. **Set Environment Variables**
   - In Render dashboard, go to your service → Environment
   - Add `GROQ_API_KEY` (if using Groq API)
   - Add `GROQ_MODEL` (optional)

4. **Upload Model File**
   - Render doesn't support large files in git
   - Options:
     - **Option A**: Use a cloud storage (S3, Google Drive) and download on startup
     - **Option B**: Include model.h5 in git (if < 100MB)
     - **Option C**: Use HuggingFace model (see `app_huggingface.py`)

#### Option 2: Manual Setup

1. **Create a new Web Service** on Render
2. **Connect your GitHub repository**
3. **Configure:**
   - **Build Command**: `pip install -r requirements_huggingface.txt && pip install gunicorn`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 180 app_huggingface:app`
   - **Environment**: Python 3
   - **Health Check Path**: `/health`

4. **Set Environment Variables:**
   - `GROQ_API_KEY` (if using Groq API)
   - `GROQ_MODEL` (optional)

5. **Deploy!**

#### Model File for Render

**✅ Recommended: Use `app_huggingface.py`** (already configured)
- Models download automatically from HuggingFace on first startup
- No need to manage model files
- Multiple model options available
- Better for cloud deployment

**Alternative: Using `app.py` with TensorFlow**
If you need to use `app.py`, you have these options:

**Option 1: Download on Startup**
Add this to `app.py` before loading the model:
```python
import urllib.request
MODEL_URL = "https://your-storage.com/model.h5"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
```

**Option 2: Include in Git**
If model is < 100MB, you can commit it (not recommended for large models).

#### Render Configuration

- **Plan**: Starter (512MB RAM) works for HuggingFace models
- **Auto-Deploy**: Enabled (deploys on git push)
- **Health Check**: `/health` endpoint
- **Timeout**: 180 seconds (for model downloading and loading)
- **First Request**: May take 30-60s to download model (cached after)

#### Testing Deployment

After deployment, test your service:
```bash
# Health check
curl https://your-service.onrender.com/health

# Prediction
curl -X POST -F "image=@test.jpg" https://your-service.onrender.com/predict
```

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### Model Not Loading
- Check `model.h5` exists and is valid
- Verify TensorFlow version compatibility
- Check file permissions

### Low Memory
- Reduce batch size
- Use smaller model (MobileNetV2 is already small)
- Add more RAM or use cloud GPU

### Slow Predictions
- Use GPU if available
- Reduce image resolution
- Use model quantization

## Resources

- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)

# IOT_ML
