import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Baca data clean...")
df = pd.read_csv("loan_data_clean.csv")
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

print("Mulai training model CI...")
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X, y)

# Simpan model sebagai artefak untuk GitHub
joblib.dump(model, "model.pkl")
print("Beres! Model disave sebagai model.pkl")