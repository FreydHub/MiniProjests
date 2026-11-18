# Emotion to Meme

Image emotion classification API using FastAPI and DeepFace, with a Streamlit interface.

```bash
pip install -r requirements.txt
python run_api.py
streamlit run main_app.py
```

The API exposes /health, /emotions, /classify and /meme/{emotion}/base64.
Emotion classification uses DeepFace model weights. Meme images are optional local assets
under `memes/` and are excluded from this source distribution. Without matching images,
the API reports meme availability and classification remains the central demo.
