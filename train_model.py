import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# -----------------------
# 1. Load Data
# -----------------------
df = pd.read_csv("drugs_side_effects_drugs_com_cleaned_final.csv")
print("✅ Data loaded:", df.shape)

# -----------------------
# 2. Extract Common Side Effect Tokens
# -----------------------

def extract_symptoms(text):
    if pd.isnull(text):
        return []
    text = re.sub(r"[;\.]", ",", text.lower())
    text = re.sub(r"\s+(or|and)\s+", ",", text)
    tokens = [t.strip() for t in text.split(",")]
    tokens = [t for t in tokens if 1 <= len(t.split()) <= 3]
    return list(set(tokens))

df['side_effect_tokens'] = df['side_effects'].apply(extract_symptoms)

# Count most common side effects
all_tokens = df['side_effect_tokens'].explode()
top_tokens = all_tokens.value_counts()
top_tokens = top_tokens[top_tokens >= 10].index.tolist()

if len(top_tokens) == 0:
    raise ValueError("❌ No common side effects with at least 10 samples.")

print(f"✅ {len(top_tokens)} unique side effects being predicted.")

# Binarize labels
mlb = MultiLabelBinarizer(classes=top_tokens)
y = mlb.fit_transform(df['side_effect_tokens'])

# -----------------------
# 3. Filter Rows with At Least One Target Label
# -----------------------
mask = y.sum(axis=1) > 0
df = df.loc[mask].reset_index(drop=True)
y = y[mask]

# -----------------------
# 4. Select Features
# -----------------------
features = [
    'drug_name',
    'generic_name',
    'drug_classes',
    'brand_names',
    'medical_condition',
    'rx_otc',
    'pregnancy_category',
    'csa',
    'alcohol',
    'rating',
    'no_of_reviews'
]

df = df[features]

# Fill NA
df = df.fillna("unknown")

# -----------------------
# 5. Preprocessing Pipeline
# -----------------------
categorical = [
    'drug_name', 'generic_name', 'drug_classes',
    'brand_names', 'medical_condition', 'rx_otc',
    'pregnancy_category', 'csa', 'alcohol'
]

numerical = ['rating', 'no_of_reviews']

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical),
    ("num", "passthrough", numerical)
])

# -----------------------
# 6. Split Data
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42
)
print(f"✅ Training on {X_train.shape[0]} samples")

# -----------------------
# 7. Pipeline and Model
# -----------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# -----------------------
# 8. Evaluate
# -----------------------
y_pred = model.predict(X_test)
print("✅ Evaluation:\n")
print(classification_report(y_test, y_pred, target_names=top_tokens))

# -----------------------
# 9. Save Artifacts
# -----------------------
joblib.dump(model, "models/side_effects_model.pkl")
joblib.dump(top_tokens, "models/target_labels.pkl")
joblib.dump(features, "models/feature_names.pkl")

print("✅ Model, features, and labels saved.")