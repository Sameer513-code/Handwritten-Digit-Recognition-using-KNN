import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load the MNIST dataset
mnist = pd.read_csv('mnist_train.csv')

# 2. Split into features (X) and labels (y)
y = mnist.iloc[:, 0].values   # First column is the label
X = mnist.iloc[:, 1:].values  # Remaining columns are pixel values

# 2. Normalize pixel values (0-255 -> 0-1)
X = X/255.0  

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train KNN model (using 3 nearest neighbors)
knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)  # Use multiple CPU cores for speed
knn.fit(X_train, y_train)

# 5. Evaluate accuracy
accuracy = knn.score(X_test, y_test)
print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")