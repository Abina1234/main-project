from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, classification_report

app = Flask(__name__)

# Load the preprocessed data
train_data = pd.read_csv('Dataset\\updated_data.csv').dropna()
test_data  = pd.read_csv('Dataset\\preprocessed_test.csv').dropna()

# Preprocess the data
tfidf = TfidfVectorizer(strip_accents='unicode', stop_words='english', lowercase=True, 
                        use_idf=True, smooth_idf=True, sublinear_tf=True, max_df=0.5)
X_train = tfidf.fit_transform(train_data['clean_essay'])
y_train = train_data['final_score']
X_test = tfidf.transform(test_data['clean_essay'])

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

svm_model = SVR(kernel='poly', degree=1, C=1, epsilon=0.1)
svm_model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    essays = data['essays']
    test_essays = pd.DataFrame({'essay': essays})
    test_essays['clean_essay'] = test_essays['essay'].apply(preprocess_essay)
    X_test_essays = tfidf.transform(test_essays['clean_essay'])
    predictions = svm_model.predict(X_test_essays)
    feedbacks = []
    for i in range(len(predictions)):
        predicted_score = predictions[i]
        feedback = ""
        # Provide feedback based on the predicted score and other analysis of the essay
        if predicted_score < 2:
            feedback = "Your essay needs significant improvement. The main issue is with the organization and development of ideas."
        elif predicted_score < 3:
            feedback = "Your essay is below average. You need to work on developing your ideas and using more specific examples."
        elif predicted_score < 4:
            feedback = "Your essay is average. You need to work on using more varied sentence structures and stronger vocabulary."
        elif predicted_score < 5:
            feedback = "Your essay is above average. Good job! Work on using more varied vocabulary and more specific examples."
        else:
            feedback = "Your essay is excellent! Great job! Just make sure to use varied vocabulary and specific examples."
        feedbacks.append(feedback)
    return jsonify(feedbacks)

def preprocess_essay(essay):
    # Add any preprocessing steps here
    return essay

if __name__ == '__main__':
    app.run(debug=True)
