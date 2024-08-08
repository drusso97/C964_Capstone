from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load and preprocess the data
dataset = pd.read_csv('data/bot_detection_data.csv')

# Prepare features and target variable
X = dataset[['Retweet Count', 'Mention Count', 'Follower Count', 'Verified']]
y = dataset['Bot Label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# # Perform feature scaling
# sc = StandardScaler()
# X_train[['Retweet Count', 'Mention Count', 'Follower Count']] = sc.fit_transform(X_train[['Retweet Count', 'Mention Count', 'Follower Count']])
# X_test[['Retweet Count', 'Mention Count', 'Follower Count']] = sc.transform(X_test[['Retweet Count', 'Mention Count', 'Follower Count']])

# Train the model
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Make predictions
y_pred = regressor.predict(X_test)

# Calculate accuracy for the test set
accuracy = accuracy_score(y_test, y_pred.round()) * 100
print("Accuracy:", accuracy)


@app.route('/')
def index():
    # Make predictions
    y_pred = regressor.predict(X_test)
    X_test['Prediction'] = y_pred
    X_test['Actual'] = y_test.values

    # Prepare data for display
    result_df = X_test.head(100).reset_index(drop=True)  # Show first 100 rows

    return render_template('index.html', tables=[result_df.to_html(classes='data', header="true")],
                           titles=result_df.columns.values)


if __name__ == '__main__':
    app.run(debug=True)
