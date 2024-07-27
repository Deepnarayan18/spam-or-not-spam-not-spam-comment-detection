from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['comment']
        # Convert the comment to a 2D array and transform it using the vectorizer
        data = cv.transform([comment])
        # Make prediction
        prediction = model.predict(data)
        result = 'Spam Comment' if prediction[0] == 'Spam Comment' else 'Not Spam Comment'
        return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
