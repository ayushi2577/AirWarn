import time
from sklearn.svm import SVC
from config import FOLDS
from splits import get_fold
from pipeline import load_model_table, model_columns, get_Xy, make_preprocessor

table = load_model_table()
cols = model_columns(table)

# Largest training fold = fold 4 (about 10,000 rows)
tr, va = get_fold(table, len(FOLDS) - 1)
X_tr, y_tr = get_Xy(tr)

pre = make_preprocessor(cols)
A = pre.fit_transform(X_tr[cols])
print("training matrix:", A.shape)

t = time.time()
SVC(kernel="rbf", C=1.0, gamma="scale").fit(A, y_tr)
print(f"one exact RBF fit: {time.time() - t:.1f} seconds")