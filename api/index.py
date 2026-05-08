"""
Fake News Detector — Flask API
Vercel entry point: api/index.py  (Vercel imports `app` via WSGI)
Local run: python api/index.py
"""
import os
import sys
import json
from datetime import datetime

from flask import Flask, request, jsonify, render_template

# ── Path setup — make src/ importable ─────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocessing import clean_text
from src.model import FakeNewsDetector
from src.utils import (
    save_prediction, get_history_df,
    clear_history, fetch_article_from_url,
)

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates')

# ── Lazy model loader (cached after first load) ────────────────────────────
_models: dict = {}

def get_model(model_type: str = 'logistic_regression') -> FakeNewsDetector | None:
    if model_type not in _models:
        try:
            _models[model_type] = FakeNewsDetector.load(
                model_type,
                directory=os.path.join(ROOT, 'models'),
            )
        except FileNotFoundError:
            _models[model_type] = None
    return _models[model_type]


# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body JSON: { "text": "...", "model": "logistic_regression" | "naive_bayes" }
    Returns:   { prediction, confidence, prob_fake, prob_real, top_words, error? }
    """
    data       = request.get_json(force=True, silent=True) or {}
    raw_text   = data.get('text', '').strip()
    model_type = data.get('model', 'logistic_regression')

    if not raw_text:
        return jsonify({'error': 'No text provided.'}), 400

    detector = get_model(model_type)
    if detector is None:
        return jsonify({
            'error': 'Model not found. Run python train.py first and redeploy.'
        }), 503

    cleaned = clean_text(raw_text)
    if len(cleaned.split()) < 3:
        return jsonify({'error': 'Text too short after preprocessing.'}), 400

    prediction, confidence, prob_dict = detector.predict(cleaned)
    top_words = detector.get_top_features(cleaned, n=12)
    save_prediction(raw_text, prediction, confidence, model_type)

    return jsonify({
        'prediction': prediction,
        'confidence': round(confidence * 100, 1),
        'prob_fake':  round(prob_dict.get('FAKE', 0) * 100, 1),
        'prob_real':  round(prob_dict.get('REAL', 0) * 100, 1),
        'top_words':  [
            {'word': w, 'score': round(s, 4), 'direction': d}
            for w, s, d in top_words
        ],
        'model': model_type,
    })


@app.route('/fetch-url', methods=['POST'])
def fetch_url():
    """
    POST /fetch-url
    Body JSON: { "url": "https://..." }
    Returns:   { title, text, word_count, error? }
    """
    data = request.get_json(force=True, silent=True) or {}
    url  = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'No URL provided.'}), 400

    title, body, err = fetch_article_from_url(url)
    if err:
        return jsonify({'error': err}), 400

    return jsonify({
        'title':      title or '',
        'text':       body,
        'word_count': len(body.split()),
        'preview':    body[:500] + ('…' if len(body) > 500 else ''),
    })


@app.route('/history', methods=['GET'])
def history():
    """GET /history — return prediction history as JSON."""
    df = get_history_df()
    if df.empty:
        return jsonify([])
    return jsonify(df.iloc[::-1].to_dict(orient='records'))


@app.route('/history', methods=['DELETE'])
def delete_history():
    """DELETE /history — clear all history."""
    clear_history()
    return jsonify({'message': 'History cleared.'})


# ── Local dev entry point ──────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
