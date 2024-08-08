import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import joblib


# Load and prepare the dataset
def load_data(file_path):
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:, 3:7]
    y = dataset.iloc[:, 7]
    return X, y


# Train and save the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    sc = StandardScaler()
    X_train.iloc[:, :3] = sc.fit_transform(X_train.iloc[:, :3])
    X_test.iloc[:, :3] = sc.transform(X_test.iloc[:, :3])
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)
    # Save the model and scaler
    joblib.dump(regressor, 'model.pkl')
    joblib.dump(sc, 'scaler.pkl')
    return regressor, sc, X_test, y_test


# Predict function
def predict(model, scaler, X):
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)
