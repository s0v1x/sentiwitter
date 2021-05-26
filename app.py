from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from clean import *
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from xgboost import XGBClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


wrod2vec = Word2Vec.load("word2vec_senti.model")
clf = XGBClassifier()
clf.load_model("model.json")
clf._le = LabelEncoder().fit(['1', '0'])
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        comment = request.form['comment']
        data = clean_tweet(comment)
        vect = docvec(data,wrod2vec)

        my_prediction = clf.predict(np.array(vect).reshape((1,-1)))
    return render_template('result.html', prediction=int(''.join(my_prediction)))


if __name__ == '__main__':
    app.run(debug=True)