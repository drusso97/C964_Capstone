from flask import Flask, request, render_template
from models.model import predict

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_bot():
    # Extract data from the form
    tweet_content = request.form['tweet_content']
    account_creation_date = request.form['account_creation_date']
    follower_count = request.form['follower_count']
    retweet_count = request.form['retweet_count']
    verification_status = request.form['verification_status']

    # Convert input data to the format expected by the model
    input_data = [tweet_content, account_creation_date, follower_count, retweet_count, verification_status]
    prediction = predict(input_data)

    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
