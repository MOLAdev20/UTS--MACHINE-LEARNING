import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score,
    roc_curve, roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt

# === BACA DATA ===
df = pd.read_excel("student-mat.xlsx")

# === PILIH KOLOM YANG DIPAKAI ===
selected_columns = [
    'sex', 'age', 'studytime', 'failures', 'schoolsup', 
    'famsup', 'goout', 'health', 'absences', 'G1', 'G2', 'G3'
]
df = df[selected_columns]

# === ENCODING DAN TARGET ===
df = pd.get_dummies(df, drop_first=True)
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
X = df.drop(columns=['G3', 'pass'])
y = df['pass']

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === NORMALISASI ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === MODEL 1: Logistic Regression ===
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

# === MODEL 2: Decision Tree ===
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

# === Evaluasi Teks ===
print("===== LOGISTIC REGRESSION =====")
print("Akurasi:", accuracy_score(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

print("===== DECISION TREE =====")
print("Akurasi:", accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# === CONFUSION MATRIX (Visual) ===
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_log)).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Logistic Regression - Confusion Matrix")

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_tree)).plot(ax=axes[1], colorbar=False)
axes[1].set_title("Decision Tree - Confusion Matrix")
plt.tight_layout()
plt.show()

# === ROC CURVE ===
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)

auc_log = roc_auc_score(y_test, y_prob_log)
auc_tree = roc_auc_score(y_test, y_prob_tree)

plt.figure(figsize=(7,5))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {auc_log:.2f})")
plt.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC = {auc_tree:.2f})")
plt.plot([0,1], [0,1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Perbandingan Model")
plt.legend()
plt.show()

# === PRECISION-RECALL CURVE ===
prec_log, rec_log, _ = precision_recall_curve(y_test, y_prob_log)
prec_tree, rec_tree, _ = precision_recall_curve(y_test, y_prob_tree)

plt.figure(figsize=(7,5))
plt.plot(rec_log, prec_log, label="Logistic Regression")
plt.plot(rec_tree, prec_tree, label="Decision Tree")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# === PERBANDINGAN AKURASI & AUC DALAM BAR CHART ===
models = ["Logistic Regression", "Decision Tree"]
accuracy_scores = [
    accuracy_score(y_test, y_pred_log),
    accuracy_score(y_test, y_pred_tree)
]
auc_scores = [auc_log, auc_tree]

plt.figure(figsize=(7,4))
plt.bar(models, accuracy_scores, color="skyblue", label="Accuracy")
plt.bar(models, auc_scores, color="orange", alpha=0.6, label="AUC")
plt.ylabel("Score")
plt.title("Perbandingan Accuracy & AUC")
plt.legend()
plt.show()
