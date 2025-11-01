import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === 1. LOAD DATA ===
df = pd.read_excel("student-mat.xlsx")

# Encode kategorikal (kalau belum)
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['schoolsup'] = df['schoolsup'].map({'yes': 1, 'no': 0})
df['famsup'] = df['famsup'].map({'yes': 1, 'no': 0})

# === 2. BIKIN TARGET KATEGORIKAL ===
df['pass_fail'] = np.where(df['G3'] >= 10, 1, 0)  # 1 = lulus, 0 = tidak lulus

# === 3. PISAH FITUR DAN TARGET ===
X = df.drop(columns=['G3', 'pass_fail'])
y = df['pass_fail']

# === 4. SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. STANDARDISASI ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 6. MODEL 1: LOGISTIC REGRESSION ===
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)

# === 7. MODEL 2: DECISION TREE ===
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# === 8. EVALUASI ===
print("===== LOGISTIC REGRESSION =====")
print("Akurasi:", accuracy_score(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

print("\n===== DECISION TREE =====")
print("Akurasi:", accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
