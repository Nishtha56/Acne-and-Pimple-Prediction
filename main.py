import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# -------------------------
# Read dataset
# -------------------------
data = pd.read_csv("Acne_Dataset.csv")
print("Dataset Loaded Successfully!\n")

# Remove extra spaces in column names
data.columns = data.columns.str.strip()

data.columns=['Age', 'Gender', 'Skin_Type', "Current_Acne", "Is_Server",
              "Exposure", "Apply_Sunscreen", "Face_Wash","Water_Intake","Junk_Food",
              "Sugar_Intake","Stress","Family_History","Face_Touch","Sleeping_Hours"]
# -------------------------
# Basic EDA
# -------------------------
print("First 5 rows:\n", data.head(), "\n")
print("Shape of dataset:", data.shape, "\n")
print("Information of dataset:\n")
print(data.info(), "\n")
print("Statistical summary:\n", data.describe(include='all'), "\n")
print("Null value count:\n", data.isnull().sum(), "\n")

# -------------------------
# Handling Missing Values
# -------------------------
num_col = data.select_dtypes(include="number").columns
cat_col = data.select_dtypes(include="object").columns

# Fill numeric missing values with mean
data[num_col] = data[num_col].fillna(data[num_col].mean())

# Fill categorical missing values with mode
for col in cat_col:
    data[col] = data[col].fillna(data[col].mode()[0])

print("Null values after cleaning:\n", data.isnull().sum(), "\n")

# -------------------------
# Label Encoding (convert text → numbers)
# -------------------------
le = LabelEncoder()
for col in cat_col:
    data[col] = le.fit_transform(data[col])

print("\nLabel Encoding Completed.")

# ---------------------------------------------
# SCALING NUMERIC COLUMNS
# ---------------------------------------------
scaler = StandardScaler()
data[num_col] = scaler.fit_transform(data[num_col])
print("\nNumeric columns scaled successfully.")

# -------------------------
# Outlier Detection (IQR method) on numeric features
# -------------------------
for col in num_col:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = data[(data[col] < lower) | (data[col] > upper)]
    print(f"Outliers in {col}: {len(outliers)}")

print("\nOutlier check completed!\n")

# -------------------------
# Outlier Visualization (Boxplots)
# -------------------------
for col in num_col:
    plt.figure(figsize=(6,4))
    sns.boxplot(y=data[col])
    plt.title(f"Outlier Detection using Boxplot for {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Lifestyle Factors vs Acne")
plt.tight_layout()
plt.show()
# -------------------------
# Train-Test Split
# -------------------------
# Detect acne target column automatically
target = [c for c in data.columns if "acne" in c.lower()][0]


# ===========================================================
#            FEATURE ENGINEERING (INCREASE CORRELATION)
# ===========================================================

# Convert Acne → Binary Target
data["Current_Acne"] = data["Current_Acne"].replace({1: 1, 2: 1, 0: 0})

# Lifestyle Risk Score
data["Lifestyle_Risk"] = (
    data["Junk_Food"] +
    data["Sugar_Intake"] +
    data["Stress"] +
    data["Sleeping_Hours"]
)

# Skincare Routine Score
data["Skincare_Score"] = (
    data["Apply_Sunscreen"] +
    data["Face_Wash"] +
    data["Face_Touch"]
)

# Hydration Score
data["Hydration_Level"] = data["Water_Intake"] + data["Stress"]

# Sun Exposure Risk
data["Sun_Exposure_Risk"] = data["Exposure"] + data["Apply_Sunscreen"]

print("\nFeature Engineering Completed!")

scaler = StandardScaler()
data[num_col] = scaler.fit_transform(data[num_col])
# -------------------------
# Analysis: Feature Importance (Objective 1)
# -------------------------



# 1. Correlation Matrix
# -------------------------
plt.figure(figsize=(12, 10))
# FIXED: Use 'data' instead of 'data_encoded'
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Lifestyle Factors vs Acne")
plt.tight_layout()
plt.show()


# 2. Random Forest Feature Importance
# -------------------------
# Drop the target and 'Severity' (since severity is a result, not a predictor)
# FIXED: Use 'data' instead of 'data_encoded'
# Features

# ------------------------------------------------------------
# OBJECTIVE 1
# ------------------------------------------------------------
X = data.drop(columns=[target])
y = data[target]

# Train model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# Extract importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Display results
print("\nTop 10 Key Factors Influencing Acne:\n")
print(importance_df.head(5))

# Plot
plt.figure(figsize=(10,6))
sns.barplot(
    x=importance_df["Importance"].head(5),
    y=importance_df["Feature"].head(5)
)
plt.title("Top Factors Influencing Acne")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


#objectuve 2

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


print("\n================ OBJECTIVE 2 MODELS ================\n")

# ------------------------------------------------------------
# NOW CREATE TRAIN/TEST SPLIT (THE FIX)
# ------------------------------------------------------------
X = data.drop(columns=["Current_Acne"])
y = data["Current_Acne"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

print("Train-Test Split Completed!\n")


# ------------------------------------------------------------
# OBJECTIVE 2: CLASSIFICATION MODELS
# ------------------------------------------------------------
scaler = StandardScaler()

# Fit on TRAIN, transform on TEST (Prevent data leakage)
X_train = scaler.fit_transform(x_train)
Y_test = scaler.transform(x_test)
print("\n--- Objective 2: Model Evaluation ---")

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}
results = []
best_model = None
best_acc = 0
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    results.append({
        'Model': name, 
        'Accuracy': f"{acc*100:.2f}%", 
        'F1-Score': f"{f1:.4f}",
        'Precision': f"{prec:.4f}",
        'Recall': f"{rec:.4f}"
    })
    
    if acc > best_acc:
        best_acc = acc
        best_model = model # Save best model for risk scoring later

results_df = pd.DataFrame(results)
print(results_df)

print("\nAll Objective 2 Models Trained Successfully!")

# ============================================================
# OBJECTIVE 3: MULTI-CLASS CLASSIFICATION (SEVERITY PREDICTION)
# ============================================================
print("\n" + "="*50)
print("OBJECTIVE 3: ACNE SEVERITY CLASSIFICATION")
print("="*50)

# 1. Reload Data to get original Severity Levels (0, 1, 2, etc.)
# ------------------------------------------------------------
df_severity = pd.read_csv("Acne_dataset.csv")
df_severity.columns = df_severity.columns.str.strip()
df_severity.columns = ['Age', 'Gender', 'Skin_Type', "Current_Acne", "Is_Server",
                       "Exposure", "Apply_Sunscreen", "Face_Wash","Water_Intake","Junk_Food",
                       "Sugar_Intake","Stress","Family_History","Face_Touch","Sleeping_Hours"]

# Handle missing values
num_col = df_severity.select_dtypes(include="number").columns
cat_col = df_severity.select_dtypes(include="object").columns
df_severity[num_col] = df_severity[num_col].fillna(df_severity[num_col].mean())
for col in cat_col:
    df_severity[col] = df_severity[col].fillna(df_severity[col].mode()[0])

# Label Encode Categorical Columns
le = LabelEncoder()
for col in cat_col:
    df_severity[col] = le.fit_transform(df_severity[col])

# 2. Prepare Features (X) and Target (y)
# ------------------------------------------------------------
# Ensure Target is Integer for Multi-class
y_multi = df_severity["Current_Acne"].astype(int) 
X_multi = df_severity.drop(columns=["Current_Acne"])

# Scale Features
scaler = StandardScaler()
X_multi = scaler.fit_transform(X_multi)

# Train-Test Split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

# 3. Train Multi-class Models
# ------------------------------------------------------------
from sklearn.metrics import classification_report

# Model A: Random Forest (Handles multi-class naturally)
rf_multi = RandomForestClassifier(n_estimators=200, random_state=42)
rf_multi.fit(X_train_m, y_train_m)
y_pred_rf = rf_multi.predict(X_test_m)

print(f"\n--- Random Forest Multi-class Report ---")
print(classification_report(y_test_m, y_pred_rf))

# Model B: Support Vector Machine (One-vs-Rest for multi-class)
svc_multi = SVC(decision_function_shape='ovr', kernel='rbf', random_state=42)
svc_multi.fit(X_train_m, y_train_m)
y_pred_svc = svc_multi.predict(X_test_m)

print(f"\n--- SVM Multi-class Report ---")
print(classification_report(y_test_m, y_pred_svc))

# Confusion Matrix Visualization for Random Forest
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test_m, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Acne Severity Prediction")
plt.xlabel("Predicted Severity")
plt.ylabel("Actual Severity")
plt.show()
# ============================================================
# OBJECTIVE 4: UNSUPERVISED CLUSTERING (SKIN PROFILES)
# ============================================================
print("\n" + "="*50)
print("OBJECTIVE 4: CLUSTERING (SKIN HEALTH PROFILES)")
print("="*50)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Use the scaled feature set (X_multi) from the previous step
X_cluster = X_multi 

# 1. Elbow Method to find optimal 'K'
# ------------------------------------------------------------
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# 2. Apply K-Means (Assume K=3 based on typical patterns, or adjust based on plot)
# ------------------------------------------------------------
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster)

# Add cluster labels to original dataframe for analysis
df_severity['Cluster_Labels'] = clusters

print("\nCluster Distribution:\n", df_severity['Cluster_Labels'].value_counts())

# 3. Visualize Clusters using PCA (2D Projection)
# ------------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=100)
plt.title(f'Skin Health Profiles (K-Means Clustering, K={optimal_k})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()



# ============================================================
# OBJECTIVE 5: ASSOCIATION RULE MINING
# ============================================================
# Ensure Current_Acne is numeric
# Objective 5: Association Patterns (Apriori)

# Convert target column safely
# Use the correct dataframe name
# ---------- DATA TYPE FIX (MANDATORY) ----------
print("\n" + "="*50)
print("OBJECTIVE 5: ASSOCIATION PATTERNS (APRIORI)")
print("="*50)

# ------------------------------------------------------------
# Import Apriori Libraries
# ------------------------------------------------------------
from mlxtend.frequent_patterns import apriori, association_rules

# ------------------------------------------------------------
# Step 1: Prepare Binary Dataset for Association Rules
# Apriori works ONLY on Boolean (True/False) data
# ------------------------------------------------------------
df_assoc = pd.DataFrame()

# ---------- FIX: Ensure Current_Acne is NUMERIC ----------
data['Current_Acne'] = data['Current_Acne'].astype(str).str.strip()
data['Current_Acne'] = pd.to_numeric(data['Current_Acne'], errors='coerce')
data['Current_Acne'] = data['Current_Acne'].fillna(0).astype(int)

# ---------- Create Binary Features ----------
df_assoc['High_Acne'] = data['Current_Acne'] >= 1

df_assoc['High_Sugar_Intake'] = (
    data['Sugar_Intake'] > data['Sugar_Intake'].mean()
)

df_assoc['High_Junk_Food'] = (
    data['Junk_Food'] > data['Junk_Food'].mean()
)

df_assoc['Low_Water_Intake'] = (
    data['Water_Intake'] < data['Water_Intake'].mean()
)

df_assoc['High_Stress'] = (
    data['Stress'] > data['Stress'].median()
)

df_assoc['Poor_Sleep'] = (
    data['Sleeping_Hours'] < data['Sleeping_Hours'].mean()
)

# Convert all columns to Boolean
df_assoc = df_assoc.astype(bool)

print("\nBinary Transaction Dataset (First 5 Rows):")
print(df_assoc.head())

# ------------------------------------------------------------
# Step 2: Apply Apriori Algorithm
# ------------------------------------------------------------
frequent_itemsets = apriori(
    df_assoc,
    min_support=0.10,   # appears in at least 10% of records
    use_colnames=True
)

print("\nFrequent Itemsets Found:")
print(frequent_itemsets.head())

# ------------------------------------------------------------
# Step 3: Generate Association Rules
# ------------------------------------------------------------
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.1
)

# Sort rules by Lift (strength)
rules = rules.sort_values(by="lift", ascending=False)

print("\nTop Association Rules:")
print(
    rules[['antecedents', 'consequents',
           'support', 'confidence', 'lift']].head(10)
)

# ------------------------------------------------------------
# Step 4: Visualization of Association Rules
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=rules['support'],
    y=rules['confidence'],
    size=rules['lift'],
    hue=rules['lift'],
    sizes=(30, 200),
    palette='viridis'
)

plt.title("Association Rules: Support vs Confidence (Size = Lift)")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()

plt.show()
