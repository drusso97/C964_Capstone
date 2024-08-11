import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import io
import base64

app = Flask(__name__)

# Load and preprocess the data
dataset = pd.read_csv('data/bot_detection_data.csv')
X = dataset[['Retweet Count', 'Mention Count', 'Follower Count', 'Verified']]
y = dataset['Bot Label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the model
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)


@app.route('/')
def index():
    # Default settings
    results_per_page = int(request.args.get('results_per_page', 25))
    page = int(request.args.get('page', 1))

    # Create a DataFrame for X_test with only the feature columns
    X_test_features = X_test[['Retweet Count', 'Mention Count', 'Follower Count', 'Verified']].copy()

    # Make predictions
    y_pred = regressor.predict(X_test)
    y_pred = y_pred.round().astype(int)  # Ensure predictions are 0 or 1

    # # Calculate accuracy for the test set
    # accuracy = round((accuracy_score(y_test, y_pred.round()) * 100), 2)
    # print("Accuracy:", accuracy)

    # Add predictions and actual values for display
    X_test_features['Prediction'] = y_pred
    X_test_features['Actual'] = y_test.values

    # Paginate results
    start_index = (page - 1) * results_per_page
    end_index = start_index + results_per_page
    paginated_df = X_test_features[start_index:end_index].reset_index(drop=True)

    # Prepare data for display
    tables = [paginated_df.to_html(classes='data', header="true", index=False)]
    titles = paginated_df.columns.values

    # Calculate total pages
    total_pages = (len(X_test_features) + results_per_page - 1) // results_per_page

    return render_template(
        'index.html',
        tables=tables,
        titles=titles,
        current_page=page,
        total_pages=total_pages,
        results_per_page=results_per_page
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        retweet_count = float(request.form['retweet_count'])
        mention_count = float(request.form['mention_count'])
        follower_count = float(request.form['follower_count'])
        verified = int(request.form['verified'])

        # Prepare input for the model
        input_data = pd.DataFrame({
            'Retweet Count': [retweet_count],
            'Mention Count': [mention_count],
            'Follower Count': [follower_count],
            'Verified': [verified]
        })

        # Make prediction
        prediction = regressor.predict(input_data)
        prediction = int(round(prediction[0]))

        return render_template('predict.html', prediction=prediction)

    return render_template('predict.html')


@app.route('/feature_importance_plot')
def get_feature_importance_plot():
    # Generate the feature importance plot
    plt.figure(figsize=(10, 8))
    feature_importances = regressor.feature_importances_
    features = X.columns
    plt.barh(features, feature_importances)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Plot')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('feature_importance_plot.html', plot_url=img_base64)


@app.route('/decision_tree_plot')
def get_decision_tree_plot():
    # Reduce the depth of the tree to speed up rendering
    regressor = DecisionTreeRegressor(max_depth=3, random_state=0)
    regressor.fit(X, y)

    # Generate the decision tree plot
    plt.figure(figsize=(15, 10))  # Adjust figure size for faster rendering
    plot_tree(regressor, feature_names=X.columns, filled=True, rounded=True)
    plt.title('Decision Tree Visualization')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()  # Close the plot to free up memory

    return render_template('decision_tree_plot.html', plot_url=img_base64)


@app.route('/visualization3')
def visualization3():
    return render_template('visualization3.html')


if __name__ == '__main__':
    app.run(debug=True)
