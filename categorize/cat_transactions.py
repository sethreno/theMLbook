import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import sys
import joblib
import os

# Check if the model already exists
model_path = 'transaction_model.joblib'
if not os.path.exists(model_path):
    print("Model not found, training based on data.csv.")

    # Load the data
    data = pd.read_csv('data.csv')

    # Preprocess the data
    data = data.dropna(subset=['Merchant', 'Category', 'Original Statement'])
    data['Text'] = data['Merchant'] + ' ' + data['Original Statement']
    X = data['Text']
    y = data['Category']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with a TfidfVectorizer and a LogisticRegression model
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1_000))

    # Define hyperparameters to tune
    parameters = {
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'logisticregression__C': [0.1, 1, 10]
    }

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best parameters found by grid search:")
    print(grid_search.best_params_)

    # Get the best model
    model = grid_search.best_estimator_

    # Save the model to disk
    joblib.dump(model, model_path)

    # Predict the categories for the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
else:
    print("Model already exists. Loading the model from disk.")

# Function to predict the category of a new purchase
def predict_category(merchant, original_statement):
    # Load the model from disk
    model = joblib.load(model_path)
    text = merchant + ' ' + original_statement
    return model.predict([text])[0]

# Read merchant name and original statement from command line
if len(sys.argv) > 1:
    merchant = sys.argv[1]

    if len(sys.argv) > 2:
        original_statement = sys.argv[2]
    else:
        original_statement = ""

    predicted_category = predict_category(merchant, original_statement)
    print(f'The predicted category for {merchant} with statement "{original_statement}" is {predicted_category}')
else:
    print("Please provide a merchant name and original statement as command line arguments.")
