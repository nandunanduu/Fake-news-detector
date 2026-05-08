"""
Prediction history helpers + URL article fetching.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd

# Vercel's filesystem is read-only — use /tmp when running on Vercel
if os.environ.get('VERCEL'):
    HISTORY_FILE = '/tmp/prediction_history.json'
else:
    HISTORY_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'prediction_history.json')


def load_history() -> List[Dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_prediction(text: str, prediction: str, confidence: float, model_type: str) -> Dict:
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    history = load_history()

    entry = {
        'timestamp':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'text_preview': (text[:120] + '…') if len(text) > 120 else text,
        'prediction':   prediction,
        'confidence':   round(confidence * 100, 2),
        'model':        model_type,
    }
    history.append(entry)

    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    return entry


def get_history_df() -> pd.DataFrame:
    history = load_history()
    if not history:
        return pd.DataFrame(columns=['timestamp', 'text_preview', 'prediction', 'confidence', 'model'])
    return pd.DataFrame(history)


def clear_history() -> None:
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


def fetch_article_from_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Fetch and extract article text from a URL.

    Returns:
        (title, body_text, error_message)
        On success: (title_or_None, article_text, None)
        On failure: (None, None, error_string)
    """
    import re

    # Basic URL sanity check
    if not re.match(r'https?://', url.strip(), re.IGNORECASE):
        return None, None, "URL must start with http:// or https://"

    # ── Try trafilatura first (best for news) ──────────────────────────
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            meta = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                output_format='json',
                with_metadata=True,
            )
            if meta:
                import json as _json
                data  = _json.loads(meta)
                title = data.get('title', '')
                body  = data.get('text', '')
                if body and len(body.split()) > 20:
                    return title or None, (f"{title}\n\n{body}".strip() if title else body), None
    except Exception:
        pass

    # ── Fallback: requests + BeautifulSoup ────────────────────────────
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        }
        resp = requests.get(url.strip(), headers=headers, timeout=12)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        # Remove noise elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header',
                         'aside', 'form', 'noscript', 'iframe']):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else None

        # Prefer <article> tag, then <main>, then all <p>
        container = soup.find('article') or soup.find('main')
        if container:
            paragraphs = container.find_all('p')
        else:
            paragraphs = soup.find_all('p')

        body = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
        body = re.sub(r'\s+', ' ', body).strip()

        if len(body.split()) < 20:
            return None, None, "Could not extract enough text from the page. The site may block bots."

        full = f"{title}\n\n{body}".strip() if title else body
        return title, full, None

    except Exception as exc:
        return None, None, f"Failed to fetch URL: {exc}"
