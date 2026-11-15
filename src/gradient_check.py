"""
Sanity check: logistic regression by hand vs scikit-learn.

What this does (it adds no new result and changes nothing else):
  1. Takes the TRAINING rows only (2015-2018). The 2019 test year is not touched.
  2. Runs the same preprocessing as the real model (fitted on those training rows).
  3. Trains logistic regression with plain gradient descent written in NumPy,
     using the cross-entropy loss and its gradient derived below.
  4. Trains scikit-learn's LogisticRegression (L2, C=1.0, same as models.py).
  5. Compares the two: loss, predicted probabilities and yes/no decisions.

Maths (y is 0 or 1, p = sigmoid(X w + b)):
  Bernoulli likelihood for one row:   p^y * (1-p)^(1-y)
  Negative log-likelihood (cross-entropy):  -[ y*log(p) + (1-y)*log(1-p) ]
  Gradient of the mean loss w.r.t. w:   (1/n) * X^T (p - y)
  Gradient w.r.t. b:                    (1/n) * sum(p - y)
  scikit-learn minimises   C * sum(loss) + 0.5*||w||^2   (intercept not penalised).
  Dividing by C*n gives the mean-loss form used below:
        mean(loss) + ||w||^2 / (2*C*n)    ->   gradient_w = X^T (p - y)/n + w/(C*n)
Run from the project root:  python src/gradient_check.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from splits import get_split
from pipeline import load_model_table, model_columns, get_Xy, make_preprocessor

C = 1.0            # same light L2 penalty as models.make_logreg
LEARNING_RATE = 0.5
ITERATIONS = 20000


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cross_entropy(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def objective(X, y, w, b):
    """Mean cross-entropy + L2 penalty (same objective sklearn minimises, divided by C*n)."""
    n = len(y)
    return cross_entropy(y, sigmoid(X @ w + b)) + (w @ w) / (2 * C * n)


def gradient_descent(X, y, lr=LEARNING_RATE, iters=ITERATIONS):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(iters):
        p = sigmoid(X @ w + b)
        grad_w = X.T @ (p - y) / n + w / (C * n)   # cross-entropy gradient + L2 term
        grad_b = np.mean(p - y)                    # intercept is not penalised
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


if __name__ == "__main__":
    table = load_model_table()
    cols = model_columns(table)
    train = get_split(table, "train")              # 2015-2018 only, never the test year
    X_df, y = get_Xy(train)
    y = y.values.astype(float)

    prep = make_preprocessor(cols)                 # fitted on training rows only
    X = prep.fit_transform(X_df[cols])
    print(f"training rows: {X.shape[0]}, features after preprocessing: {X.shape[1]}")

    w, b = gradient_descent(X, y)
    p_hand = sigmoid(X @ w + b)

    sk = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    p_sk = sk.predict_proba(X)[:, 1]

    print(f"\nobjective, hand-written GD : {objective(X, y, w, b):.6f}")
    print(f"objective, scikit-learn    : {objective(X, y, sk.coef_[0], sk.intercept_[0]):.6f}")
    print(f"max |p_hand - p_sklearn|   : {np.max(np.abs(p_hand - p_sk)):.4f}")
    print(f"mean |p_hand - p_sklearn|  : {np.mean(np.abs(p_hand - p_sk)):.5f}")
    same = np.mean((p_hand >= 0.5) == (p_sk >= 0.5))
    print(f"same yes/no decision at 0.5: {same:.2%} of rows")
    print(f"correlation of the weights : {np.corrcoef(w, sk.coef_[0])[0, 1]:.4f}")
    print("\nIf the objectives and probabilities agree closely, the hand-written "
          "gradient is correct and matches what scikit-learn does inside.")