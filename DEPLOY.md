# Quick Render Deployment Guide

## Prerequisites
- GitHub repository with your code
- Render account (free tier available)
- Model file (`model.h5`) - see options below

## Quick Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for Render"
   git push origin main
   ```

2. **Deploy on Render**
   - Go to https://dashboard.render.com
   - Click "New +" → "Blueprint"
   - Connect your GitHub repo
   - Render auto-detects `render.yaml`

3. **Set Environment Variables** (in Render dashboard)
   - `GROQ_API_KEY` - Your Groq API key (if using)
   - `GROQ_MODEL` - Optional Groq model name

4. **Model File** ✅ Already Configured!
   
   - Using `app_huggingface.py` (already set in `render.yaml`)
   - Models download automatically from HuggingFace
   - No model file management needed!

   **Alternative: Download on Startup**
   - Upload `model.h5` to cloud storage (S3, Google Drive, etc.)
   - Add download code to `app.py` before `load_model()`

5. **Deploy!**
   - Render will build and deploy automatically
   - Service URL: `https://your-service.onrender.com`

## Testing

```bash
# Health check
curl https://your-service.onrender.com/health

# Prediction
curl -X POST -F "image=@test.jpg" https://your-service.onrender.com/predict
```

## Troubleshooting

- **Build fails**: Check `requirements.txt` for all dependencies
- **Model not found**: Ensure model file is available (see options above)
- **Timeout**: Increase timeout in `render.yaml` (already set to 120s)
- **Memory issues**: Upgrade to Standard plan (1GB RAM)

## Notes

- Free tier: 512MB RAM, may sleep after inactivity
- Model downloading takes ~30-60 seconds on first startup (cached after)
- Use health check endpoint to keep service awake
- Using HuggingFace models (app_huggingface.py) - no local model file needed!

