import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

#LOAD MULTIPLE FILES FROM A DIRECTORY
print("📂 Please make sure all CSV files are in the 'data' folder (or change the path accordingly)")
data_folder = "E:\\AIML-vip\\bank"  # Replace with the absolute path of the folder where all CSVs are stored
file_list = [file for file in os.listdir(data_folder) if file.endswith('.csv')]

if not file_list:
    raise ValueError("❌ No CSV files found in the specified directory.")

dataframes = []
for filename in file_list:
    file_path = os.path.join(data_folder, filename)
    print(f"📥 Reading: {filename}")
    try:
        df = pd.read_csv(file_path, sep=';')
        dataframes.append(df)
    except Exception as e:
        print(f"❌ Could not read {filename}: {e}")

#VERIFY AND COMBINE
base_columns = dataframes[0].columns.tolist()
for df in dataframes:
    if df.columns.tolist() != base_columns:
        raise ValueError("❌ Uploaded files do not have the same structure.")

df = pd.concat(dataframes, ignore_index=True)
print(f"\n✅ Combined Dataset Shape: {df.shape}")
print("\n🔍 Sample Data:\n", df.head())

#CHECK FOR MISSING VALUES
print("\n🚫 Null Values:")
print(df.isnull().sum())

#TARGET VALUE DISTRIBUTION
print("\n🎯 Target Distribution:\n", df['y'].value_counts())

#ENCODE CATEGORICAL DATA
print("\n🧠 Encoding categorical variables...")
df_encoded = pd.get_dummies(df.drop('y', axis=1), drop_first=True)
df_encoded['y'] = df['y'].map({'yes': 1, 'no': 0})
print("✅ Encoding complete.")
print(f"📐 Encoded Shape: {df_encoded.shape}")

#SPLIT DATA
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n📊 Train Set:", X_train.shape)
print("📊 Test Set:", X_test.shape)

#TRAIN DECISION TREE
print("\n🌳 Training the Decision Tree model...")
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("✅ Training complete.")

#PREDICT AND EVALUATE
y_pred = model.predict(X_test)
print("\n📈 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#PLOT DECISION TREE
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True)
plt.title("🌳 Decision Tree")
plt.show()

#CONFUSION MATRIX HEATMAP
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("📊 Confusion Matrix Heatmap")
plt.show()

# PREDICT SAMPLE
sample = X_test.iloc[[0]]
prediction = model.predict(sample)[0]
print("\n🔎 Single Prediction:")
print("➡️ Predicted:", "Yes" if prediction == 1 else "No")
print("✅ Actual:", "Yes" if y_test.iloc[0] == 1 else "No")
