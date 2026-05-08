"""
Download (or generate) a dataset, preprocess it, and train both models.

Usage:
    python train.py
"""
import os
import sys
import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import preprocess_dataframe
from src.model import FakeNewsDetector


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def download_dataset() -> pd.DataFrame:
    """Try HuggingFace datasets first; fall back to a built-in sample."""
    print("Attempting to load dataset from HuggingFace …")
    try:
        from datasets import load_dataset  # type: ignore
        ds  = load_dataset("GonzaloA/fake_news", split="train")
        df  = pd.DataFrame(ds)
        df  = df[['title', 'text', 'label']].copy()
        # label: 0 = FAKE, 1 = REAL
        df['label']     = df['label'].map({0: 'FAKE', 1: 'REAL'})
        df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        print(f"  ✓ Loaded {len(df):,} articles from HuggingFace.")
        return df
    except Exception as exc:
        print(f"  ✗ HuggingFace download failed ({exc}).")

    # Try local CSV files (place fake.csv / true.csv in data/)
    fake_path = os.path.join('data', 'Fake.csv')
    real_path = os.path.join('data', 'True.csv')
    if os.path.exists(fake_path) and os.path.exists(real_path):
        print("  → Using local Fake.csv / True.csv …")
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
        fake_df['label'] = 'FAKE'
        real_df['label'] = 'REAL'
        df = pd.concat([fake_df, real_df], ignore_index=True)
        text_col = 'text' if 'text' in df.columns else df.columns[1]
        title_col = 'title' if 'title' in df.columns else df.columns[0]
        df['full_text'] = df[title_col].fillna('') + ' ' + df[text_col].fillna('')
        print(f"  ✓ Loaded {len(df):,} articles from local CSV.")
        return df

    print("  → Generating built-in demo dataset …")
    return _demo_dataset()


def _demo_dataset() -> pd.DataFrame:
    """Small balanced dataset for smoke-testing the pipeline."""
    fake = [
        "SHOCKING: Scientists discover secret mind-control chip in vaccines distributed worldwide",
        "BREAKING: Government admits moon landing was filmed in a Hollywood studio",
        "EXCLUSIVE: Drinking bleach cures cancer — suppressed by Big Pharma for decades",
        "5G towers proven to cause coronavirus, governments covering up massive conspiracy",
        "Secret society of lizard people controls all world governments, insider reveals",
        "FBI files prove Elvis Presley faked his death and lives in Hawaii under alias",
        "Bill Gates admits microchipping population through flu shots for total surveillance",
        "Miracle cure for diabetes hidden from public by pharmaceutical corporations",
        "Ancient pyramids built by aliens, newly declassified NASA documents confirm",
        "Water fluoridation is mind-control program started by secret government agency",
        "Scientists prove that eating chocolate every day reverses aging process completely",
        "CIA whistleblower reveals chemtrails contain population-control chemicals",
        "Obama secretly born in Kenya, forged birth certificate scandal exposed",
        "Mainstream media admits COVID-19 was engineered in US lab to control population",
        "Doctor confirms common household Wi-Fi causes brain cancer in children under 10",
    ] * 40

    real = [
        "Federal Reserve raises interest rates by 25 basis points to combat persistent inflation",
        "Scientists publish research on climate change effects on Arctic ice melt patterns",
        "City council approves $50 million infrastructure improvement plan for downtown roads",
        "University study finds regular exercise reduces risk of heart disease by 30 percent",
        "Tech company announces quarterly earnings surpassing analyst expectations by 12 percent",
        "International climate summit reaches agreement on carbon emission reduction targets",
        "New clinical trials show promising results for Alzheimer's disease treatment",
        "Stock market closes higher after positive employment data released by Labor Department",
        "Local government implements recycling program aiming to reduce landfill waste by 40 percent",
        "NASA successfully launches satellite to monitor global atmospheric conditions",
        "New COVID-19 variant detected in multiple countries, health officials urge vigilance",
        "Unemployment rate drops to lowest level in three years, says Labor Department report",
        "Congress passes bipartisan bill to fund renewable energy infrastructure projects",
        "WHO recommends updated flu vaccine formulations for the upcoming winter season",
        "Scientists at CERN announce breakthrough in particle physics research findings",
    ] * 40

    df = pd.DataFrame({
        'full_text': fake + real,
        'label':     ['FAKE'] * len(fake) + ['REAL'] * len(real),
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(df: pd.DataFrame):
    print("\nPreprocessing text …")
    df = preprocess_dataframe(df, text_column='full_text')
    df = df[df['cleaned_text'].str.len() > 10].reset_index(drop=True)

    X, y = df['cleaned_text'], df['label']
    print(f"  Samples : {len(X):,}")
    print(f"  FAKE    : {(y == 'FAKE').sum():,}")
    print(f"  REAL    : {(y == 'REAL').sum():,}")

    results = {}
    for model_type in ('logistic_regression', 'naive_bayes'):
        print(f"\n{'─'*50}")
        print(f"Training {model_type.replace('_', ' ').title()} …")
        detector = FakeNewsDetector(model_type=model_type)
        metrics  = detector.train(X, y)
        path     = detector.save()

        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1 Score  : {metrics['f1']:.4f}")
        print(f"  Saved to  : {path}")
        print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
        print(f"           FAKE   REAL")
        cm = metrics['confusion_matrix']
        print(f"  FAKE  {cm[0][0]:>6}  {cm[0][1]:>6}")
        print(f"  REAL  {cm[1][0]:>6}  {cm[1][1]:>6}")
        results[model_type] = metrics

    print(f"\n{'═'*50}")
    print("All models trained and saved successfully!")
    print("Run the app with:  streamlit run app.py")
    return results


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    df = download_dataset()
    train_models(df)
