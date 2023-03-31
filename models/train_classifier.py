# import libraries
import sys
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')

from sqlalchemy import create_engine
import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath).connect()
    df = pd.read_sql_table('disaster_response', engine)
    X = df.message
    Y = df[df.columns[4:]]
    category_names = Y.columns
    return X, Y, category_names 

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('cvect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
])
    parameters = {
        'clf__estimator__n_estimators': [5],
        'clf__estimator__min_samples_split': [2],
    
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=2)
    return model

def get_metrics(test_value, predicted_value):
    accuracy = accuracy_score(test_value, predicted_value)
    precision = round(precision_score(
        test_value, predicted_value, average='micro'))
    recall = recall_score(test_value, predicted_value, average='micro')
    f1 = f1_score(test_value, predicted_value, average='micro')
    return {'Accuracy': accuracy, 'f1 score': f1, 'Precision': precision, 'Recall': recall}



def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    #report = classification_report(Y_test.values, Y_pred)
                                   #, target_names=category_names)
    #print(report)
    test_results = []
    for i, column in enumerate(Y_test.columns):
        result = get_metrics(Y_test.loc[:, column].values, Y_pred[:, i])
        test_results.append(result)
    test_results_df = pd.DataFrame(test_results)
    print("Result: ")
    print(test_results_df)
    print("Evaluation Result")
    print(test_results_df.mean())

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()