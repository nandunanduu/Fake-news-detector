"""
Fake news detection model: Logistic Regression or Naive Bayes,
backed by a TF-IDF vectorizer.
"""
import os
import pickle
import numpy as np
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


class FakeNewsDetector:
    """Wraps TF-IDF + classifier with train / predict / explain helpers."""

    SUPPORTED_MODELS = ('logistic_regression', 'naive_bayes')

    def __init__(self, model_type: str = 'logistic_regression'):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")

        self.model_type = model_type
        self.is_trained = False
        self.metrics: Dict = {}

        self.vectorizer = TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000, C=1.0, solver='lbfgs'
            )
        else:
            self.model = MultinomialNB(alpha=0.1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X,
        y,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict:
        """Fit model on X/y and compute evaluation metrics on held-out split."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec  = self.vectorizer.transform(X_test)

        self.model.fit(X_train_vec, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test_vec)

        self.metrics = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label='FAKE', zero_division=0),
            'recall':    recall_score(y_test, y_pred, pos_label='FAKE', zero_division=0),
            'f1':        f1_score(y_test, y_pred, pos_label='FAKE', zero_division=0),
            'confusion_matrix':      confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']),
            'classification_report': classification_report(y_test, y_pred, zero_division=0),
            'test_size': len(y_test),
        }
        return self.metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Returns:
            prediction  – 'FAKE' or 'REAL'
            confidence  – probability of the predicted class  (0-1)
            prob_dict   – {'FAKE': p, 'REAL': p}
        """
        self._check_trained()
        vec = self.vectorizer.transform([text])
        prediction  = self.model.predict(vec)[0]
        probs       = self.model.predict_proba(vec)[0]
        prob_dict   = dict(zip(self.model.classes_, probs))
        confidence  = prob_dict[prediction]
        return prediction, confidence, prob_dict

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def get_top_features(self, text: str, n: int = 15) -> List[Tuple[str, float, str]]:
        """
        Return the top-n words in *text* that most influenced the prediction.

        Each entry: (word, score, 'FAKE' | 'REAL')
        Positive score → pushes toward REAL; negative → pushes toward FAKE.
        """
        self._check_trained()
        vec            = self.vectorizer.transform([text])
        feature_names  = self.vectorizer.get_feature_names_out()
        text_arr       = vec.toarray()[0]
        present        = np.where(text_arr > 0)[0]

        if len(present) == 0:
            return []

        if self.model_type == 'logistic_regression':
            # coef_ shape (1, n_features); positive → REAL (class index 1)
            coefs  = self.model.coef_[0]
            scores = [(feature_names[i], float(text_arr[i] * coefs[i])) for i in present]
        else:
            # Use difference of log-probs between REAL and FAKE
            classes   = list(self.model.classes_)
            real_idx  = classes.index('REAL')
            fake_idx  = classes.index('FAKE')
            log_diff  = self.model.feature_log_prob_[real_idx] - self.model.feature_log_prob_[fake_idx]
            scores    = [(feature_names[i], float(log_diff[i])) for i in present]

        scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return [(w, s, 'REAL' if s > 0 else 'FAKE') for w, s in scores[:n]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str = MODEL_DIR) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f'{self.model_type}_model.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(model_type: str = 'logistic_regression', directory: str = MODEL_DIR) -> 'FakeNewsDetector':
        path = os.path.join(directory, f'{model_type}_model.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No saved model at '{path}'. Run train.py first."
            )
        with open(path, 'rb') as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
