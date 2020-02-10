# Disaster Response Pipeline
## Udacity Data Scientist nanodegree - Project 2

### Project Objective:
Use NLP to classify messages related to disaster responses to 36 possible categorical classes. The purpose is to help first responders distinguish between true emergency messages and determine when and what type of resources are needed. A simple web app shows plots describing the dataset and an input form demonsatrates message classification to the the 36 possible classes.

There are three components to this project.
1. ETL Pipeline

process_data.py is a data cleaning pipeline that:

    Loads the 'messages' and 'categories' datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

2. ML Pipeline

train_classifier.py is a machine learning pipeline that:

    Loads data the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file

3. Flask Web App

	A simple web app where the user can write in a sentence about a natural disaster and see which type of disaster it is classified as


### File Structure:
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md


### Dependencies:
json
numpy
plotly
pandas
pickle
sys

functools partial
time
nltk
nltk.stem WordNetLemmatizer
nltk.tokenize word_tokenize

flask
flask render_template, request, jsonify
plotly.graph_objs Bar
sklearn.externals joblib
sqlalchemy create_engine

sklearn.model_selection train_test_split
sklearn.linear_model LogisticRegression
sklearn.model_selection RandomizedSearchCV, GridSearchCV
sklearn.ensemble RandomForestClassifier
sklearn.multioutput MultiOutputClassifier
sklearn.metrics classification_report
sklearn.pipeline Pipeline, FeatureUnion
sklearn.feature_extraction.text CountVectorizer, TfidfTransformer
xgboost XGBClassifier


