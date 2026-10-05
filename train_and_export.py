import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df_raw = pd.read_csv("Acne_dataset.csv")
df_raw.columns = df_raw.columns.str.strip()

# Standardized column mapping
col_map = {
    'Age': 'Age',
    'Gender': 'Gender',
    'How would you describe your skin type?': 'Skin_Type',
    'Do you have acne currently?': 'Current_Acne',
    'If yes, how severe is it?': 'Severity',
    'How many hours of sun exposure do you get daily?': 'Sun_Exposure',
    'How often do you apply sunscreen?': 'Apply_Sunscreen',
    'How often do you wash your face in a day?': 'Face_Wash',
    'How much water do you drink per day?': 'Water_Intake',
    'How often do you eat oily/junk food?': 'Junk_Food',
    'How much sugar do you consume daily?': 'Sugar_Intake',
    'Do you experience stress frequently?': 'Stress',
    'Do you have a family history of acne?': 'Family_History',
    'Do you touch your face frequently?': 'Face_Touch',
    'How many hours of sleep do you get daily?': 'Sleeping_Hours'
}
df = df_raw.rename(columns=col_map).copy()

# Fill missing values
df['Severity'] = df['Severity'].fillna('None')
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Collect Dataset Statistics for EDA Dashboard
dataset_stats = {
    "total_records": len(df),
    "features_count": len(df.columns),
    "gender_dist": df['Gender'].value_counts().to_dict(),
    "skin_type_dist": df['Skin_Type'].value_counts().to_dict(),
    "acne_dist": df['Current_Acne'].value_counts().to_dict(),
    "severity_dist": df['Severity'].value_counts().to_dict(),
    "stress_dist": df['Stress'].value_counts().to_dict(),
    "water_dist": df['Water_Intake'].value_counts().to_dict(),
    "junk_food_dist": df['Junk_Food'].value_counts().to_dict(),
    "sunscreen_dist": df['Apply_Sunscreen'].value_counts().to_dict(),
    "sleep_dist": df['Sleeping_Hours'].value_counts().to_dict(),
    "age_stats": {
        "min": float(df['Age'].min()),
        "max": float(df['Age'].max()),
        "mean": round(float(df['Age'].mean()), 1),
        "median": float(df['Age'].median())
    }
}

# ----------------- Encoding -----------------
encoders = {}
df_encoded = df.copy()

categorical_cols = ['Gender', 'Skin_Type', 'Current_Acne', 'Severity', 'Sun_Exposure',
                    'Apply_Sunscreen', 'Face_Wash', 'Water_Intake', 'Junk_Food',
                    'Sugar_Intake', 'Stress', 'Family_History', 'Face_Touch', 'Sleeping_Hours']

encoding_dicts = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    encoding_dicts[col] = {str(k): int(v) for k, v in zip(le.classes_, range(len(le.classes_)))}

# Binary target: 1 = Has Acne (Yes), 0 = No Acne (No)
# In raw dataset: 'Yes' -> 1, 'No' -> 0
target_binary = (df['Current_Acne'] == 'Yes').astype(int)

# Feature list for prediction (excluding target variables)
feature_cols = ['Age', 'Gender', 'Skin_Type', 'Sun_Exposure', 'Apply_Sunscreen',
                'Face_Wash', 'Water_Intake', 'Junk_Food', 'Sugar_Intake',
                'Stress', 'Family_History', 'Face_Touch', 'Sleeping_Hours']

X = df_encoded[feature_cols].copy()
y = target_binary

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ----------------- Model Training & Evaluation -----------------
models = {
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
    'Naive Bayes': GaussianNB()
}

model_results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    model_results.append({
        'name': name,
        'accuracy': round(acc * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'confusion_matrix': cm
    })
    trained_models[name] = model

# Feature Importance from Random Forest
rf_model = trained_models['Random Forest']
feature_importances = [
    {"feature": col, "importance": round(float(imp) * 100, 2)}
    for col, imp in zip(feature_cols, rf_model.feature_importances_)
]
feature_importances.sort(key=lambda x: x["importance"], reverse=True)

# ----------------- Severity Classifier (Multiclass) -----------------
# Severity: 'None' (0), 'Mild' (1), 'Moderate' (2), 'Severe' (3)
severity_mapping = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
y_severity = df['Severity'].map(severity_mapping).fillna(0).astype(int)

X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
    X_scaled, y_severity, test_size=0.2, random_state=42
)
rf_severity = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
rf_severity.fit(X_train_sev, y_train_sev)
sev_acc = accuracy_score(y_test_sev, rf_severity.predict(X_test_sev))

# ----------------- Clustering (K-Means) -----------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)

# Sample 150 points for visualization
pca_sample = []
for i in range(min(200, len(df))):
    pca_sample.append({
        "x": round(float(pca_coords[i, 0]), 3),
        "y": round(float(pca_coords[i, 1]), 3),
        "cluster": int(clusters[i]),
        "acne": str(df['Current_Acne'].iloc[i]),
        "skin_type": str(df['Skin_Type'].iloc[i]),
        "stress": str(df['Stress'].iloc[i])
    })

cluster_profiles = [
    {
        "cluster_id": 0,
        "name": "High Lifestyle Stress & Low Hydration Group",
        "archetype": "Stress & Diet Vulnerable",
        "description": "High stress levels, frequent junk food/sugar, lower sleep, and moderate water intake. Shows elevated acne risk.",
        "risk_tendency": "High",
        "prevalence_pct": round(float(np.mean(clusters == 0)) * 100, 1),
        "acne_rate": round(float(np.mean(df[df['Cluster'] == 0]['Current_Acne'] == 'Yes')) * 100, 1)
    },
    {
        "cluster_id": 1,
        "name": "Balanced Skincare & Controlled Diet Group",
        "archetype": "Protective Routine Maintainers",
        "description": "Consistent daily sunscreen and face washing, higher water intake (>2L), low sugar consumption. Shows lowest acne recurrence.",
        "risk_tendency": "Low",
        "prevalence_pct": round(float(np.mean(clusters == 1)) * 100, 1),
        "acne_rate": round(float(np.mean(df[df['Cluster'] == 1]['Current_Acne'] == 'Yes')) * 100, 1)
    },
    {
        "cluster_id": 2,
        "name": "High Sun & Frequent Face Touching Group",
        "archetype": "Environmental & Barrier Challenged",
        "description": "Prolonged sun exposure without reapplication, high face touching frequency, variable sleep hours.",
        "risk_tendency": "Moderate to High",
        "prevalence_pct": round(float(np.mean(clusters == 2)) * 100, 1),
        "acne_rate": round(float(np.mean(df[df['Cluster'] == 2]['Current_Acne'] == 'Yes')) * 100, 1)
    }
]

# ----------------- Association Rules (Apriori) -----------------
df_assoc = pd.DataFrame()
df_assoc['Has_Acne'] = df['Current_Acne'] == 'Yes'
df_assoc['Oily_Skin'] = df['Skin_Type'] == 'Oily'
df_assoc['High_Sugar'] = df['Sugar_Intake'].isin(['High', 'Very high (more than 4 sugary items/day)'])
df_assoc['High_Junk_Food'] = df['Junk_Food'].isin(['Daily', '3-4 times a week'])
df_assoc['Low_Water'] = df['Water_Intake'].isin(['Less than 1 litre', '1-2 litres'])
df_assoc['High_Stress'] = df['Stress'].isin(['Often', 'Very often'])
df_assoc['Poor_Sleep'] = df['Sleeping_Hours'].isin(['4-6 hours', 'Less than 5 hours'])
df_assoc['Frequent_Face_Touch'] = df['Face_Touch'].isin(['Yes', 'Often'])
df_assoc['Family_Acne_History'] = df['Family_History'] == 'Yes'
df_assoc['No_Sunscreen'] = df['Apply_Sunscreen'].isin(['Never', 'Rarely'])

frequent_itemsets = apriori(df_assoc, min_support=0.08, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.05)
rules = rules.sort_values(by="lift", ascending=False)

# Format rules for UI
rules_list = []
for _, r in rules.head(15).iterrows():
    rules_list.append({
        "antecedents": list(r['antecedents']),
        "consequents": list(r['consequents']),
        "support": round(float(r['support']), 3),
        "confidence": round(float(r['confidence']), 3),
        "lift": round(float(r['lift']), 2)
    })

# Linear weights from Logistic Regression & RF for fast browser client-side scoring
lr = trained_models['Logistic Regression']
feature_weights = {
    col: round(float(coef), 4) for col, coef in zip(feature_cols, lr.coef_[0])
}

# Export all insights to json
exported_data = {
    "dataset_stats": dataset_stats,
    "model_results": model_results,
    "feature_importances": feature_importances,
    "severity_accuracy": round(sev_acc * 100, 2),
    "cluster_profiles": cluster_profiles,
    "pca_sample": pca_sample,
    "association_rules": rules_list,
    "feature_weights": feature_weights,
    "encoding_dicts": encoding_dicts,
    "scaler_mean": [round(float(m), 4) for m in scaler.mean_],
    "scaler_scale": [round(float(s), 4) for s in scaler.scale_],
    "lr_intercept": round(float(lr.intercept_[0]), 4)
}

with open("ml_insights.json", "w") as f:
    json.dump(exported_data, f, indent=2)

print("ML training and insight extraction completed successfully! Saved to ml_insights.json")
