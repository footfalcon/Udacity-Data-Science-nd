import sys
import numpy as np
import pandas as pd
import pickle
from functools import partial
from sqlalchemy import create_engine
import time
# nltk imports
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from xgboost import XGBClassifier

def load_data(database_filepath):
    ''' Reads in the 'messages' table from the 'DisasterResponse' SQLite db and output 
        the messages as the X vector and the 36 classification category targets as the 
        y matrix.

        Args:
        - database_filepath (str): path to the sqlite databse
        - param_testing (int): integer on where to slice the df to reduce size for tuning paramaters

        Returns: X vecotr of messages, y matrix of category targets
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)
    X = df.message.values
    y = df.iloc[:, 3: ].values
    category_names = df.columns[3:]

    return X, y, category_names


def tokenize(text):
    ''' Takes in a message, tokenizes into lowercase words and lemmatizes into lemmas.

        Args:
        - text (str): message string

        Returns (list): cleaned word tokens of the message
    '''
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words('english')]  
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(t, pos='n').strip() for t in tokens]
    #clean_tokens = [lemmatizer.lemmatize(w, pos='v').strip() for w in clean_tokens]  # do both?
    # what about stemming?? Stemming increases recall while harming precision?

    return clean_tokens


def build_model():
    ''' Takes in the message column as input and output classification results on the 
        other 36 categories in the dataset.
    '''
    # to view parameters: pipeline.get_params()
    pipeline = Pipeline([
        ('vect', CountVectorizer(
            tokenizer=partial(tokenize), 
            max_df=0.52,
            max_features=3000)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(XGBClassifier(
            n_jobs=8,
            learning_rate=0.2,
            max_depth=1,
            n_estimators=400
            )))
    ])

    parameters = {
            #'vect__ngram_range': ((1, 1), (1, 5)),
            #'vect__max_df': (0.52, 0.55),
            #'vect__max_features': (1900, 3000),
            #'tfidf__use_idf': (True, False),
            #'tfidf__ngram_range': ((1, 1), (1, 5)),
            #'clf__estimator__n_estimators': [280, 320],
            #'clf__estimator__max_depth': [180, 225],
            #'clf__estimator__n_estimators': [390, 440],
            #'clf__estimator__max_depth': [1, 2],
            #'clf__estimator__learning_rate': [0.2, 0.25]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=1)
    '''cv = RandomizedSearchCV(
        pipeline, 
        param_distributions=parameters, 
        n_jobs=8, 
        cv=3, 
        verbose=1,
        random_state=0
    )'''

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' Prints the model results from 'classification_report' which includes
        precision, recall and f1-score.

        Args:
        - model (GridSearchCV): trained model
        - X_test (array): array of disaster text message/sentences
        - y_test (array): array of binaries of 36 possible disaster responses classes
        - category_names (list): names of 36 classes

        Returns (dict): dict of results (and prints results and best model parameters)
    '''
    y_pred = model.predict(X_test)
    results = classification_report(Y_test, y_pred, target_names=category_names) #, output_dict=True)
    print(results)
    print()
    print(f"\nBest Parameters: {model.best_params_}")
    #results = pd.DataFrame(results).T

    return None #results


def save_model(model, model_filepath):
    ''' Saves trained model as a pickle file

        Args:
        - model (GridSearchCV): trained model
        - model_filepath (str): dir where model is saved

        Returns: None (just saves the pickled file)
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

        
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