"""
Training script for the sentiment classifier.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from classifier.model import build_pipeline
from classifier.preprocess import clean_text


def train(texts: list, labels: list, test_size: float = 0.2):
    """Train and evaluate the sentiment classifier."""
    cleaned = [clean_text(t) for t in texts]
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=test_size, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds))

    return pipeline
