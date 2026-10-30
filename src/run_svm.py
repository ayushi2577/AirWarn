from models import *

table = load_model_table()
cols = model_columns(table)

print("Tuning linear SVM...")
lin_grid = [{"C": c} for c in [0.01, 0.1, 1.0]]
lin = tune(table, make_linear_svm, cols, lin_grid)
print(lin.round(4).to_string(index=False), "\n")

print("Tuning RBF SVM...")
rbf_grid = [{"C": c, "gamma": g} for c in [0.5, 1.0, 4.0] for g in [0.003, 0.01, 0.03]]
rbf = tune(table, make_rbf_svm, cols, rbf_grid)
print(rbf.round(4).to_string(index=False), "\n")