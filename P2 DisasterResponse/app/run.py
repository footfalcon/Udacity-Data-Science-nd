import json
import plotly
import pandas as pd

import nltk
nltk.download(['stopwords'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# create graphs
def get_graphs(df=df):
    ''' Create graphs for web app

        Args:
        - df: dataframe created from the sqlite db
        
        Returns: graphs (list) - list of plotly graph objects
    '''
    # extract data for plot 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # plot
    graph1 = [Bar(x=genre_names,  y=genre_counts, opacity=0.6)]
    layout1 = {
        'yaxis': {'title': "Count"},
        'xaxis': {'title': "Genre"}
    }

    # extract data for plot 2
    class_counts = df.iloc[:, 3:].sum().sort_values(ascending=False)
    class_names = list(class_counts.index)
    # plot
    graph2 = [Bar(x=class_names,  y=class_counts)]
    layout2 = {
        'yaxis': {'title': "Count"},
        'xaxis': {'title': "Class"}
    }
    
    # extract data for plot 3
    text = []
    for sent in df.message:
        text.extend(sent.lower().split(' '))
    stop = stopwords.words('english')
    stop.extend(['people', 'said', 'one', 'will', 'bit', 'ly', 'co', 'RT', ',', '-', '.', '..', ''])
    text = [i for i in text if i not in stop]
    word_counts = pd.Series(text).value_counts()
    word_names = list(word_counts.index)
    # plot
    graph3 = [Bar(x=word_names,  y=word_counts[:30])]
    layout3 = {
        'yaxis': {'title': "Count"},
        'xaxis': {'title': "Word"}
    }    
    
    # extract data for plot 4
    # extract data
    # data names
    # plot
    '''graph4 = [Bar(x=word_names,  y=word_counts[:30])]
    layout4 = {
        'yaxis': {'title': "Count"},
        'xaxis': {'title': "Word"}
    } '''     
    graphs = []
    graphs.append(dict(data=graph1, layout=layout1))
    graphs.append(dict(data=graph2, layout=layout2))
    graphs.append(dict(data=graph3, layout=layout3))
    #graphs.append(dict(data=graph4, layout=layout4))

    return graphs


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # get graphs
    graphs = get_graphs(df)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # get graphs
    graphs = get_graphs(df)
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template('go.html', query=query, ids=ids, classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()