# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 02:47:16 2019

@author: sukandulapati
"""

import pandas as pd
import pickle
import re
from keras.engine.saving import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

meta_data = pickle.load(open("gsuite_v1", "rb"))

intent_category = meta_data['intent_category']
#X_train = meta_data['X_train']
#y_train = meta_data['y_train']
word_index = meta_data['word_index']
max_len = meta_data['max_len']
emb_dim = meta_data['emb_dim']
tokenizer = meta_data['tokenizer']
#train_data = meta_data['train_data']
response_data = pd.read_excel('responsedata.xlsx', 'Sheet1')

stop_words = stopwords.words('english')
def clean_up_sentence(str_object, stop_words):
    words = []

    # Remove punctuations
    text = re.sub('[^a-zA-Z-0-9]', ' ', str_object)

    # Convert to lowercase
    text = text.lower()

    # remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    # remove special characters and digits
    text = re.sub("(\\d-|\\W)+", " ", text)

    # Stemming
    # text = stem_text(text.split()).split()
    text = text.split()
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in stop_words]
    words.extend(text)
    text = " ".join(text)
    #    sentence_words = sorted(list(set(words)))
    return text

#from keras.models import load_model
loaded_model = load_model('gsuite_v1.h5')


#
def get_prediction(query, recommendations=3):
    cleaned_question = clean_up_sentence(query, stop_words)
    query = [cleaned_question]
    seq = tokenizer.texts_to_sequences(query)
    padded = pad_sequences(seq, maxlen=max_len)
    pred = loaded_model.predict(padded)[0]
    pred_df = pd.DataFrame(pred, intent_category)
    pred_df.columns = ['probability']
    pred_df['Intent'] = pred_df.index
    pred_df = pred_df.sort_values('probability',ascending=False)
    results = pred_df[:recommendations].values.tolist()
    return results
#
get_prediction(query='hi')


while(1):
    user_input = input('user->')
    prediction=get_prediction(user_input)
    print('prediction socre: ' + str(prediction[0][0]) + '  ' + 'intent category: ' + str(prediction[0][1]))
    print('\nResponse: ')
    print(response_data[response_data.intent_name == prediction[0][1]]['response'].item())

input('user->')














































# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class prediction(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        query = str(user_query)
        if 'limit' in request.get:
            limit = int(request.get('limit'))
        else:
            limit = 3
        stop_words = stopwords.words('english')
        cleaned_question = clean_up_sentence(query, stop_words)
        query = [cleaned_question]

        seq = tokenizer.texts_to_sequences(query)
        padded = pad_sequences(seq, maxlen=max_len)
        pred = loaded_model.predict(padded)[0]
        pred_df = pd.DataFrame(pred, intent_category)
        pred_df.columns = ['probability']
        pred_df['intent_name'] = pred_df.index
        pred_df = pred_df.sort_values('probability',ascending=False)
        results = pred_df[:limit].values.tolist()

        response_df = pd.DataFrame(columns=['intent_name', 'friendly_name', 'probability', 'response'])
        for i in range(0, limit):
            tmp_rdf = response_data[response_data['intent_name'] == results[i][1]]
            tmp_rdf['probability'] = round(results[i][0], 3)
            tmp_rdf = tmp_rdf[['intent_name', 'friendly_name', 'probability', 'response']]
            tmp_rdf = tmp_rdf.reset_index(drop=True)
            response_df = response_df.append(tmp_rdf, ignore_index=True)
            final_json = response_df.to_json(orient='records', lines=True)
        return final_json

api.add_resource(prediction, '/')

if __name__ == '__main__':
    app.run()

#
#@app.route("/", methods=["GET"])
#def default():
#    return "Debug endpoint reached"
#
#@app.route("/query", methods=["POST"])
#
#def prediction():
#    request = request.get_json()
#    print('request received')
#
#
#    query = str(request.get("query"))
#
#    if 'limit' in request.get:
#            limit = int(request.get('limit'))
#    else:
#        limit = 3
#
#    stop_words = stopwords.words('english')
#    cleaned_question = clean_up_sentence(query, stop_words)
#    query = [cleaned_question]
#
#    seq = tokenizer.texts_to_sequences(query)
#    padded = pad_sequences(seq, maxlen=max_len)
#    pred = loaded_model.predict(padded)[0]
#    pred_df = pd.DataFrame(pred, intent_category)
#    pred_df.columns = ['probability']
#    pred_df['intent_name'] = pred_df.index
#    pred_df = pred_df.sort_values('probability',ascending=False)
#    results = pred_df[:limit].values.tolist()
#
#    response_df = pd.DataFrame(columns=['intent_name', 'friendly_name', 'probability', 'response'])
#    for i in range(0, limit):
#        tmp_rdf = response_data[response_data['intent_name'] == results[i][1]]
#        tmp_rdf['probability'] = round(results[i][0], 3)
#        tmp_rdf = tmp_rdf[['intent_name', 'friendly_name', 'probability', 'response']]
#        tmp_rdf = tmp_rdf.reset_index(drop=True)
#        response_df = response_df.append(tmp_rdf, ignore_index=True)
#        final_json = response_df.to_json(orient='records', lines=True)
#    return final_json
#
#
#if __name__ == "__main__":
#     app.run(port=6080)
##    app.run(host="0.0.0.0", port=6080, debug=True, threaded=True)  # for linux
#    # app.run(port=6080)  # for windows
