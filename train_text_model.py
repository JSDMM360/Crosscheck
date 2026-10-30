"""Train the optional text model used by the writing-style branch.

    python train_text_model.py --csv data.csv --text-col text --label-col label [--group-col source]

The label column may be 1/0, or strings such as fake/real. Datasets like ISOT, LIAR or FakeNewsNet work
once exported to CSV. Pass --group-col (e.g. the publisher) so the test split holds out whole sources:
random splits on these datasets are known to overstate accuracy because models learn publisher quirks.
Output: models/text_clf.joblib (load only files you trained yourself: joblib files are pickles).
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline

from fnd.config import MODELS_DIR

FAKE_WORDS = {"fake", "false", "1", "true_fake", "misinformation", "pants-fire", "barely-true"}


def to_binary(v) -> int:
    return int(str(v).strip().lower() in FAKE_WORDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col")
    ap.add_argument("--max-rows", type=int, default=200_000)
    a = ap.parse_args()

    df = pd.read_csv(a.csv).dropna(subset=[a.text_col, a.label_col]).head(a.max_rows)
    X, y = df[a.text_col].astype(str).str.slice(0, 4000), df[a.label_col].map(to_binary)
    if a.group_col:
        tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(X, y, df[a.group_col]))
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    else:
        print("Warning: random split. Use --group-col to hold out whole sources.")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    base = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True),
                         LogisticRegression(max_iter=1000, class_weight="balanced"))
    cv = max(2, min(5, int(ytr.value_counts().min())))
    # probability calibration matters: the fusion layer treats this output as a real probability
    model = CalibratedClassifierCV(base, method="sigmoid", cv=cv).fit(Xtr, ytr)

    p = model.predict_proba(Xte)[:, 1]
    print(f"test rows {len(yte)} | F1 {f1_score(yte, p >= .5):.3f} | AUROC {roc_auc_score(yte, p):.3f} "
          f"| Brier {brier_score_loss(yte, p):.3f}")
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "text_clf.joblib")
    print("saved", MODELS_DIR / "text_clf.joblib")


if __name__ == "__main__":
    main()
