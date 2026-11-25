"""
Multilabel classification of occupational disease categories from free text
Turkish occupational and clinical histories.

This script reproduces the analysis pipeline described in:

  Beyan AC, Eyuboglu AK. Classifying Occupational Disease Categories from
  Free Text Turkish Case Histories of Mining and Metal Industry Workers with
  Multilabel Natural Language Processing. BMC Medical Informatics and
  Decision Making.

The clinical records themselves are NOT included in this repository because
they contain sensitive occupational health information. The code expects a
de-identified, leakage-controlled corpus prepared as described in the paper
and in the accompanying README.

Expected input file (CSV or XLSX) with one row per worker and the columns:
    input_text            leakage-controlled pre-diagnosis narrative
    label_pnomokonyoz     0/1   pneumoconiosis
    label_isitme          0/1   occupational noise-induced hearing loss
    label_kasiskelet      0/1   occupational musculoskeletal disorders
    label_dermatit        0/1   occupational dermatitis

Author: Ali Kemal Eyuboglu
License: MIT
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_recall_fscore_support, f1_score,
                             roc_auc_score, accuracy_score, hamming_loss,
                             brier_score_loss)
from sklearn.model_selection import train_test_split, KFold
from scipy.sparse import hstack

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

LABELS = ["label_pnomokonyoz", "label_isitme", "label_kasiskelet", "label_dermatit"]
LABEL_NAMES = ["Pneumoconiosis", "Noise-induced hearing loss",
               "Musculoskeletal disorders", "Dermatitis"]
SEED = 42


def tr_lower(text):
    if not isinstance(text, str):
        return ""
    repl = {"İ": "i", "I": "ı", "Ş": "ş", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ç": "ç"}
    for a, b in repl.items():
        text = text.replace(a, b)
    return text.lower()


def build_vectorizers():
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                               min_df=5, max_features=20000,
                               sublinear_tf=True, preprocessor=tr_lower)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                               max_features=15000, sublinear_tf=True,
                               preprocessor=tr_lower)
    return word_vec, char_vec


def fit_features(texts, word_vec, char_vec, fit=True):
    if fit:
        Xw = word_vec.fit_transform(texts)
        Xc = char_vec.fit_transform(texts)
    else:
        Xw = word_vec.transform(texts)
        Xc = char_vec.transform(texts)
    return hstack([Xw, Xc]).tocsr()


def make_logreg():
    return LogisticRegression(C=3.0, penalty="l2", class_weight="balanced",
                              solver="liblinear", max_iter=2000)


def make_xgb(pos_weight):
    return XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.12,
                         subsample=0.9, colsample_bytree=0.5,
                         scale_pos_weight=pos_weight, eval_metric="logloss",
                         random_state=SEED, n_jobs=4)


def tune_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate(Y_true, Y_prob, thresholds):
    Y_pred = np.zeros_like(Y_prob, dtype=int)
    for j in range(Y_true.shape[1]):
        Y_pred[:, j] = (Y_prob[:, j] >= thresholds[j]).astype(int)

    rows = []
    for j, name in enumerate(LABEL_NAMES):
        p, r, f, _ = precision_recall_fscore_support(
            Y_true[:, j], Y_pred[:, j], average="binary", zero_division=0)
        try:
            auc = roc_auc_score(Y_true[:, j], Y_prob[:, j])
        except ValueError:
            auc = float("nan")
        rows.append((name, p, r, f, auc))

    micro_f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    subset_acc = accuracy_score(Y_true, Y_pred)
    hl = hamming_loss(Y_true, Y_pred)
    overall = dict(micro_f1=micro_f1, macro_f1=macro_f1,
                   subset_accuracy=subset_acc, hamming_loss=hl)
    return rows, overall, Y_pred


def bootstrap_ci(Y_true, Y_prob, thresholds, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    N = Y_true.shape[0]
    micro, subset = [], []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        _, ov, _ = evaluate(Y_true[idx], Y_prob[idx], thresholds)
        micro.append(ov["micro_f1"])
        subset.append(ov["subset_accuracy"])
    def ci(v):
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return ci(micro), ci(subset)


def top_terms(model, word_vec, char_vec, k=8):
    names = np.array(word_vec.get_feature_names_out().tolist()
                     + char_vec.get_feature_names_out().tolist())
    coefs = model.coef_.ravel()
    order = np.argsort(coefs)[::-1][:k]
    return list(zip(names[order], coefs[order]))


def main(path, model_name):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df = df.dropna(subset=["input_text"]).reset_index(drop=True)
    texts = df["input_text"].astype(str).values
    Y = df[LABELS].astype(int).values

    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED)

    word_vec, char_vec = build_vectorizers()
    Xtr = fit_features(texts[tr], word_vec, char_vec, fit=True)
    Xva = fit_features(texts[va], word_vec, char_vec, fit=False)
    Xte = fit_features(texts[te], word_vec, char_vec, fit=False)

    prob_va = np.zeros((len(va), len(LABELS)))
    prob_te = np.zeros((len(te), len(LABELS)))
    models = []
    for j in range(len(LABELS)):
        ytr = Y[tr, j]
        if model_name == "xgb" and HAVE_XGB:
            pos_w = max(1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum()))
            clf = make_xgb(pos_w)
        else:
            clf = make_logreg()
        clf.fit(Xtr, ytr)
        models.append(clf)
        prob_va[:, j] = clf.predict_proba(Xva)[:, 1]
        prob_te[:, j] = clf.predict_proba(Xte)[:, 1]

    thresholds = [tune_threshold(Y[va, j], prob_va[:, j]) for j in range(len(LABELS))]

    rows, overall, _ = evaluate(Y[te], prob_te, thresholds)
    print("\n=== Per-category performance (held-out test set) ===")
    print(f"{'Category':30s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s}")
    for name, p, r, f, auc in rows:
        print(f"{name:30s} {p:6.3f} {r:6.3f} {f:6.3f} {auc:6.3f}")

    print("\n=== Overall ===")
    for k, v in overall.items():
        print(f"{k:16s}: {v:.3f}")

    print("\n=== Calibration (Brier score) ===")
    for j, name in enumerate(LABEL_NAMES):
        try:
            b = brier_score_loss(Y[te, j], prob_te[:, j])
            print(f"{name:30s}: {b:.3f}")
        except ValueError:
            pass

    micro_ci, subset_ci = bootstrap_ci(Y[te], prob_te, thresholds)
    print("\n=== Bootstrap 95% CIs ===")
    print(f"micro-F1       : {overall['micro_f1']:.3f} "
          f"({micro_ci[0]:.3f} to {micro_ci[1]:.3f})")
    print(f"subset accuracy: {overall['subset_accuracy']:.3f} "
          f"({subset_ci[0]:.3f} to {subset_ci[1]:.3f})")

    print("\n=== 5-fold cross-validation (entire modelling set) ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    micro_scores = []
    for tr_i, te_i in kf.split(idx):
        tr_i2, va_i2 = train_test_split(tr_i, test_size=0.1765, random_state=SEED)
        wv, cv = build_vectorizers()
        Xa = fit_features(texts[tr_i2], wv, cv, True)
        Xv = fit_features(texts[va_i2], wv, cv, False)
        Xt = fit_features(texts[te_i], wv, cv, False)
        pv = np.zeros((len(va_i2), len(LABELS)))
        pt = np.zeros((len(te_i), len(LABELS)))
        for j in range(len(LABELS)):
            clf = make_logreg()
            clf.fit(Xa, Y[tr_i2, j])
            pv[:, j] = clf.predict_proba(Xv)[:, 1]
            pt[:, j] = clf.predict_proba(Xt)[:, 1]
        th = [tune_threshold(Y[va_i2, j], pv[:, j]) for j in range(len(LABELS))]
        _, ov, _ = evaluate(Y[te_i], pt, th)
        micro_scores.append(ov["micro_f1"])
    print(f"micro-F1: {np.mean(micro_scores):.3f} (SD {np.std(micro_scores):.3f})")

    if model_name != "xgb":
        print("\n=== Top terms per category (logistic regression coefficients) ===")
        for j, name in enumerate(LABEL_NAMES):
            terms = top_terms(models[j], word_vec, char_vec, k=8)
            pretty = ", ".join(f"{t}" for t, _ in terms)
            print(f"{name}: {pretty}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the prepared corpus (CSV or XLSX)")
    ap.add_argument("--model", default="logreg", choices=["logreg", "xgb"])
    args = ap.parse_args()
    main(args.data, args.model)
