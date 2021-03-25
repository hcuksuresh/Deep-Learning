# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 20:59:03 2019

@author: sukandulapati
"""

#import xlrd
import pandas as pd
#import logging
import pickle
import random
import re
#import threading
#import warnings
#from datetime import datetime

import nltk
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense
from keras.layers import Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_excel('intents_gsuite.xlsx', 'Sheet1')
df.columns = ['intent_name', 'phrase', 'response']
df['friendly_name'] = df.intent_name

#try to split data into train and test, apply augmentation only for train data

intent_category = sorted(list(df.intent_name.unique()))
# augment utterances based on words
def augment(sentence, n):
    new_sentences = []
    words = word_tokenize(sentence)
    for i in range(n):
        random.shuffle(words)
        new_sentences.append(' '.join(words))
    new_sentences = list(set(new_sentences))
    return new_sentences

utterances_df = pd.DataFrame(columns=['phrase', 'intent_name'])
for i in range(len(intent_category)):
    single_intent_df = df[df['intent_name'] == intent_category[i]]
    utterances_list = []
    for utterance in single_intent_df['phrase']:
        sentenses = augment(utterance, 5)
        utterances_list.extend(sentenses)
    temp_df = pd.DataFrame(utterances_list, columns=['phrase'])
    temp_df['intent_name'] = single_intent_df['intent_name'].unique()[0]
    utterances_df = utterances_df.append(temp_df, ignore_index=True)

stop_words = stopwords.words('english')

# create a data structure to hold user context
def nlp_task(phrase_series_object, stop_words):
    corpus = []
    words = []
    word_count = []
    for i in range(len(phrase_series_object)):
        text = re.sub('[^a-zA-Z0-9]', ' ', phrase_series_object[i])
        text = text.lower()
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
        text = re.sub("(\\d-|\\W)+", " ", text)
        text = text.split()
        # text = stem_text(text.split()).split()
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in stop_words]
        words.extend(text)
        word_count.append(len(text))
        text = " ".join(text)
        corpus.append(text)
        words = sorted(list(set(words)))
    return corpus, words, word_count

utterances_df['corpus'], words, utterances_df['word_count'] = nlp_task(utterances_df['phrase'], stop_words)

#max_len = utterances_df.word_count.max()
max_len = 12

concated = utterances_df[utterances_df.word_count != 0]

concated = concated.groupby('intent_name',as_index = False,group_keys=False).apply(lambda s: s.sample(40,replace=True))

# assigning a numeric label for each intent category
concated['LABEL'] = concated['intent_name']

# Initializing one-hot encode for each numeric category
le = preprocessing.LabelEncoder()
le.fit(concated['LABEL'])
concated['LABEL'] = le.transform(concated.LABEL)

# just checking the original numeric category
#print(concated['LABEL'][:10])

# convering LABEL to one-hot encode vector
labels = to_categorical(concated['LABEL'], num_classes=len(concated.LABEL.unique()))
# print(labels[:10])
if 'CATEGORY' in concated.keys():
    concated.drop(['CATEGORY'], axis=1)

# just by observing the unique tokens in the corpus
n_most_common_words = len(words)

# initializing the tokenizer with specified common words and max len of the sentence
tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

# fitting the tokenizer to the data
tokenizer.fit_on_texts(concated['corpus'].values)

# converting tokens to sequence
sequences = tokenizer.texts_to_sequences(concated['corpus'].values)

# word indexingz
#word_index = tokenizer.word_index
word_index = words
# Just printing the number of unique words
# print('Found %s unique tokens.' % len(word_index))
n_most_common_words = len(word_index)

# pading the sequence based on the max. len
X = pad_sequences(sequences, maxlen=max_len)


# spliting the data train and test
X_train, X_test, y_train, y_test = train_test_split(X, labels, stratify = labels, test_size=0.2)




# initializing the parameters
epochs = 100
emb_dim = 64
batch_size = 16


# building the model
model = Sequential()
model.add(Embedding(n_most_common_words, emb_dim, input_length=X_train.shape[1], name = 'w1'))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(64, dropout=0.6, activation='relu', recurrent_dropout=0.4, name = 'w2'))
model.add(Dense(y_train.shape[1], activation='softmax', name = 'output'))
optimizer = Adam(lr=0.001, beta_1=0.991, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

# just checking the model summary
# print(model.summary())

# defining the checkpoint with model name to save at each incrimental accuracy
checkpoint = [ModelCheckpoint('gsuite_v1.h5', save_best_only=True), TensorBoard()]

# model.fit(train_x, train_y, epochs=1000, batch_size=batch_size,
#      validation_split=0.2, callbacks=[LearningRateScheduler(lr_schedule),
#        ModelCheckpoint("models/model_d4096_d4096_d4096.h5", save_best_only=True),
#        TensorBoard()])
# training and tracking the history of the training
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                    callbacks=checkpoint)

# evaluating the model and printing the test accuracy
accr = model.evaluate(X_train, y_train)
print('Train set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))


accr = model.evaluate(X_test, y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))



def back_index(x):
    return np.argmax(x)

y_test_original = [] #y-test
for i in range(0, len(y_test)):
    y_test_original.append(back_index(y_test[i]))


pred_test = model.predict_classes(X_test)
print(classification_report(y_test_original, pred_test, target_names=intent_category))
final_test_result = classification_report(y_test_original, pred_test, target_names=intent_category)
#final_result = classification_report(y_test_original, pred_test)
print(final_test_result)


 # saving the required training data
pickle.dump({'intent_category': intent_category, 'X_train': X_train, 'y_train': y_train, 'word_index': word_index,
             'max_len': max_len, 'emb_dim': emb_dim, 'tokenizer': tokenizer, 'train_data': df}, open("gsuite_v1", "wb"))
