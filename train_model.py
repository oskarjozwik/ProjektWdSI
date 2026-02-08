import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import joblib
import json

# Konfiguracja cech
USED_FEATURES = ['habitat', 'population', 'cap-shape', 'cap-color', 'odor', 'gill-size', 'gill-color', 'stalk-shape', 'ring-number']

df = pd.read_csv('mushrooms.csv')

# Enkodowanie
encoders = {}
df_encoded = df.copy()
for column in df.columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df[column])
    encoders[column] = le

X = df_encoded[USED_FEATURES]
y = df_encoded['class']

# Podział 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# trening
model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# statystyki na zbiorze testowym
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred).tolist()
data_dist = df['class'].value_counts().to_dict() # e: jadalne, p: trujące
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}") #doklanosc
stats = {
    "confusion_matrix": cm,
    "data_dist": data_dist
}
print("statystyki:")
print(stats)
# Zapisywanie
joblib.dump(model, 'mushroom_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
with open('model_stats.json', 'w') as f:
    json.dump(stats, f)

print("Model i statystyki zapisane!")