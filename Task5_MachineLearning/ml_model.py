import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# =============================
# 1️⃣ Load Dataset
# =============================
df = pd.read_csv("titanic.csv")

print("Dataset Shape:", df.shape)

# =============================
# 2️⃣ Handle Missing Values
# =============================
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# =============================
# 3️⃣ Feature Engineering
# =============================
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# =============================
# 4️⃣ Select Features
# =============================
features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone']
X = df[features]
y = df['Survived']

categorical = ['Sex','Embarked']
numerical = ['Pclass','Age','Fare','FamilySize','IsAlone']

# =============================
# 5️⃣ Preprocessing Pipeline
# =============================
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
])

# =============================
# 6️⃣ Train/Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =============================
# 7️⃣ Models
# =============================
models = {
    "Logistic Regression": Pipeline([
        ('prep', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

# =============================
# 8️⃣ Train & Evaluate
# =============================
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n==========================")
    print(name)
    print("==========================")
    print("Accuracy:", round(acc,4))
    print("F1 Score:", round(f1,4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

# =============================
# 9️⃣ Feature Importance
# =============================
rf_model = models["Random Forest"]
rf_model.fit(X_train, y_train)

feature_names = numerical + list(
    rf_model.named_steps['prep']
    .named_transformers_['cat']
    .get_feature_names_out(categorical)
)

importances = rf_model.named_steps['model'].feature_importances_

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

print("\nMachine Learning Pipeline Completed ✅")

input("Press Enter to exit...")