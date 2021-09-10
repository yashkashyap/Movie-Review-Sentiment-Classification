from flask import Flask, render_template, request, url_for
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import *

import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

model_st = SentenceTransformer('nli-roberta-base')
model = load_model('moviereview.h5')

def cleanText(text):
    STOPWORDS = nltk.corpus.stopwords.words('english')

    text = text.lower()
    text = re.sub(r'<([A-Za-z]+(\s)*/*)>', '', text)
    text = re.sub(r'[,.!?]', ' ', text)
    text = [word for word in text.split(' ') if word not in STOPWORDS]

    return ' '.join(text)

def processText(text):
    text = cleanText(text)
    embeddings = model_st.encode(text)
    embeddings = embeddings.reshape(1,768)
    return embeddings

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        review = request.form['review']
        # print(review)
        review_ebd = processText(review)
        pred_prob = model.predict(review_ebd)[0][0]
        # pred_prob = 0.6
        prediction = 'Positive' if pred_prob >= 0.5 else 'Negative'
        # print(prediction)
        result = {'prediction' : prediction, 'review' : review, 'pred_prob' : pred_prob if prediction is 'Positive' else 1-pred_prob}
        return render_template('home.html', result=result)
    else:
        return render_template('home.html')

@app.route('/about/')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=False)
