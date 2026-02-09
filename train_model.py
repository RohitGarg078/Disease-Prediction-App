import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("Dataset.csv")

# Clean columns & values
df.columns = [c.strip().lower() for c in df.columns]
df = df.apply(lambda col: col.str.strip().str.lower() if col.dtype == "object" else col)
df = df.drop_duplicates()

# Target
y = df["disease"]

# Symptoms
symptom_cols = [c for c in df.columns if c.startswith("symptom")]
df["symptoms"] = df[symptom_cols].values.tolist()
df["symptoms"] = df["symptoms"].apply(
    lambda row: [s for s in row if pd.notna(s) and s != ""]
)

# Encode symptoms
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptoms"])

# Encode disease
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)

# Model
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Accuracy
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Test  Accuracy: {test_acc*100:.2f}%")

# =======================
# 📊 Accuracy Graph
# =======================
plt.figure()
plt.bar(["Train", "Test"], [train_acc, test_acc])
plt.title("Train vs Test Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# =======================
# 🔥 Confusion Matrix
# =======================
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)
disp.plot(xticks_rotation=90)
plt.title("Confusion Matrix")
plt.show()

# =======================
# 📈 Top 15 Important Symptoms
# =======================
importances = model.feature_importances_
indices = importances.argsort()[-15:]

plt.figure()
plt.barh(
    [mlb.classes_[i] for i in indices],
    importances[indices]
)
plt.title("Top 15 Important Symptoms")
plt.xlabel("Importance")
plt.show()

# Save
joblib.dump(model, "Disease_model.joblib")
joblib.dump(mlb, "Symptom_encoder.joblib")
joblib.dump(le, "Disease_encoder.joblib")

print("Model, encoders & graphs generated")
