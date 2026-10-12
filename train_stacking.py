"""
Huấn luyện mô hình STACKING phát hiện gian lận (dữ liệu PaySim - Fraud.csv).

Base learners : RandomForest (bagging) + LightGBM + XGBoost (boosting)
Meta learner  : Logistic Regression (học cách kết hợp xác suất của 3 mô hình)

Chạy:  python train_stacking.py
Cài  :  pip install pandas numpy scikit-learn lightgbm xgboost joblib
"""
import json

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import ALL_FEATURES, CAT_FEATURES, add_features

SEED = 42
NEG_POS_RATIO = 20      # giữ tối đa 20 giao dịch bình thường cho mỗi 1 giao dịch gian lận khi huấn luyện
KEEP_THRESHOLD = 0.95   # giữ các đặc trưng cho tới khi đạt 95% độ quan trọng tích lũy

# ------------------------------------------------------------------ 1. Dữ liệu
df = pd.read_csv("Fraud.csv", nrows=500000)
print("Kích thước dữ liệu:", df.shape)

df = df.dropna(subset=["isFraud"])
# orig_type / dest_type bị loại: gần như trùng thông tin với cột 'type'
df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"], errors="ignore")
df = add_features(df)

X = df[ALL_FEATURES]
y = df["isFraud"].astype(int)

# Test: giữ NGUYÊN phân phối thật, không đụng tới khi huấn luyện
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
# Validation: dùng để chọn ngưỡng quyết định + tính permutation importance
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=SEED, stratify=y_train_full)

# ------------------------------------------- 2. Cân bằng lớp (undersample lớp đa số)
# Không dùng SMOTE trên toàn bộ dữ liệu vì stacking dùng cross-validation nội bộ:
# SMOTE trước CV sẽ làm mẫu tổng hợp rò rỉ giữa các fold -> điểm số ảo.
pos_idx = y_fit[y_fit == 1].index
neg_idx = y_fit[y_fit == 0].sample(
    n=min((y_fit == 0).sum(), NEG_POS_RATIO * len(pos_idx)), random_state=SEED).index
train_idx = pos_idx.union(neg_idx)
X_bal, y_bal = X_fit.loc[train_idx], y_fit.loc[train_idx]
print(f"Tập huấn luyện sau undersample: {len(X_bal):,} dòng "
      f"(gian lận {y_bal.mean():.2%})")

# ------------------------------------------------------------ 3. Mô hình stacking
base_learners = [
    ("rf", RandomForestClassifier(
        n_estimators=200, max_depth=14, min_samples_leaf=2,
        n_jobs=-1, random_state=SEED)),
    ("lgbm", lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        n_jobs=-1, random_state=SEED, verbose=-1)),
    ("xgb", xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        eval_metric="logloss", n_jobs=-1, random_state=SEED)),
]

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    n_jobs=1,
)

prep = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
model = Pipeline([("prep", prep), ("stack", stack)])

print("Đang huấn luyện stacking (có thể mất vài phút)...")
model.fit(X_bal, y_bal)
print("Huấn luyện xong!")

# --------------------------------------------- 4. Chọn ngưỡng tối ưu F1 trên validation
val_proba = model.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, val_proba)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_thr = float(thr[np.argmax(f1s)])
print(f"Ngưỡng tối ưu (theo F1 trên validation): {best_thr:.4f}")

# ------------------------------------------------------------ 5. Đánh giá trên test
test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_thr).astype(int)
cm = confusion_matrix(y_test, test_pred)
metrics = {
    "Precision": float(precision_score(y_test, test_pred)),
    "Recall": float(recall_score(y_test, test_pred)),
    "F1-Score": float(f1_score(y_test, test_pred)),
    "AUC-ROC": float(roc_auc_score(y_test, test_proba)),
    "PR-AUC": float(average_precision_score(y_test, test_proba)),
    "threshold": best_thr,
    "confusion_matrix": cm.tolist(),
}
print(json.dumps(metrics, indent=2, ensure_ascii=False))

# --------------------------------------------------- 6. Độ quan trọng của đặc trưng
feature_names = list(model.named_steps["prep"].get_feature_names_out())
fitted_stack = model.named_steps["stack"]

# 6a. Trung bình độ quan trọng (đã chuẩn hóa) của 3 base learners
per_model = {}
for name in ["rf", "lgbm", "xgb"]:
    imp = np.asarray(fitted_stack.named_estimators_[name].feature_importances_, dtype=float)
    per_model[name] = imp / imp.sum()

imp_df = pd.DataFrame(per_model, index=feature_names)
imp_df["Importance"] = imp_df.mean(axis=1)
imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(names="Feature")
imp_df["Cumulative"] = imp_df["Importance"].cumsum()
# Giữ đặc trưng nếu trước nó tổng tích lũy chưa đạt ngưỡng
imp_df["Decision"] = np.where(
    (imp_df["Cumulative"] - imp_df["Importance"]) < KEEP_THRESHOLD, "Giữ", "Loại bỏ")
imp_df.to_csv("feature_importance.csv", index=False)

# 6b. Permutation importance của TOÀN BỘ stacking (đo đóng góp thật vào kết quả cuối)
rng = np.random.RandomState(SEED)
val_pos = X_val[y_val == 1].index
val_neg = y_val[y_val == 0].sample(n=min(30000, (y_val == 0).sum()), random_state=SEED).index
sub_idx = val_pos.union(val_neg)
perm = permutation_importance(
    model, X_val.loc[sub_idx], y_val.loc[sub_idx],
    scoring="average_precision", n_repeats=3, random_state=SEED, n_jobs=1)
perm_df = pd.DataFrame({
    "Feature": ALL_FEATURES,
    "Mean": perm.importances_mean,
    "Std": perm.importances_std,
}).sort_values("Mean", ascending=False)
perm_df.to_csv("permutation_importance.csv", index=False)

# ------------------------------------------------------------------ 7. Lưu kết quả
joblib.dump({"model": model, "threshold": best_thr}, "fraud_stacking.joblib")
with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# Xuất kết quả test để làm dashboard (nếu cần)
out = X_test.copy()
out["Actual_Fraud"] = y_test
out["Predicted_Fraud"] = test_pred
out["Fraud_Probability"] = test_proba
out.to_csv("fraud_detection_results.csv", index=False)

print("Đã lưu: fraud_stacking.joblib, metrics.json, feature_importance.csv, "
      "permutation_importance.csv, fraud_detection_results.csv")
