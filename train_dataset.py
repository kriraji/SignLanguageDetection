import os
import csv
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DATASET_PATH = "dataset"
GESTURES = ["hello", "bye", "thanks", "please", "yes", "no", "help", "sorry"]

X = []
y = []

# Load data from CSV files
for idx, gesture in enumerate(GESTURES):
    file_path = os.path.join(DATASET_PATH, f"{gesture}.csv")
    
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 63:  # 21 landmarks × 3 (x,y,z)
                X.append(list(map(float, row)))
                y.append(idx)

X = np.array(X)
y = np.array(y)

print("Total samples loaded:", len(X))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Model accuracy:", acc)


joblib.dump(model, "svm_model.pkl")
print("Model saved as svm_model.pkl")
