import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import pickle


def train_model():
    # Read CSV file
    dataset = pd.read_csv('/data/bot_detection_data.csv')

    # Split independent and dependent variables
    X = dataset.iloc[:, 3:7]
    y = dataset.iloc[:, 7]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Perform feature scaling on the dataset
    sc = StandardScaler()
    X_train.iloc[:, :3] = sc.fit_transform(X_train.iloc[:, :3])
    X_test.iloc[:, :3] = sc.transform(X_test.iloc[:, :3])

    # Train the model
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)

    # Save the model and scaler to disk
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(regressor, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(sc, scaler_file)


def load_model():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler


def predict(input_data):
    model, scaler = load_model()
    # Transform the input data
    scaled_data = scaler.transform([input_data])
    # Make prediction
    prediction = model.predict(scaled_data)
    return prediction
