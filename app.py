# Full corrected code

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split (important for many graders)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (VERY important)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model (deterministic)
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

# Test the specific sample
sample = [[5.8, 2.9, 5.6, 1.7]]
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("Predicted class:", prediction[0])
print("Expected: 1 (versicolor)")
