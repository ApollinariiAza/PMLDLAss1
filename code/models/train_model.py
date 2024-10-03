from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset and split it
wine_data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine_data.data, wine_data.target, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'C:/Users/pazar/PythonProjects/PMLDL/PMLDLAss1/code/models/wine_model.pkl')
