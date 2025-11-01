import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ===== 1. Baca dataset =====
df = pd.read_excel("student-mat.xlsx")
print("===== INFO DATA =====")
print(df.info())
print("\n===== CEK MISSING VALUE =====")
print(df.isnull().sum())

# ===== 2. Statistik deskriptif =====
print("\n===== STATISTIK DESKRIPTIF =====")
print(df.describe())

# ===== 3. Korelasi antar variabel numerik =====
df_encoded = pd.get_dummies(df, drop_first=True)

plt.figure(figsize=(12,8))
sns.heatmap(df_encoded.corr(), annot=False, cmap='coolwarm')
plt.title('Korelasi setelah encoding')
plt.tight_layout()
plt.show()


# ===== 4. Encode kolom kategorikal =====
df_encoded = pd.get_dummies(df, drop_first=True)

# ===== 5. Pisahkan fitur dan target =====
X = df_encoded.drop('G3', axis=1)
y = df_encoded['G3']

# ===== 6. Split dataset =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== 7. Normalisasi fitur numerik =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n===== DONE PREPROCESSING =====")
print(f"Jumlah data train: {len(X_train)}")
print(f"Jumlah data test: {len(X_test)}")
