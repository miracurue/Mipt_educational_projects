import pandas as pd
import re
from datetime import datetime
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from pymystem3 import Mystem
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from tqdm import tqdm



def open_file(url, num_row = 1):
    df = pd.read_json(url, lines=True)
    print(df.head(num_row))
    return df


def train_test_val(series1, series2):
    X = series1
    y = series2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    return X_train, y_train, X_test, X_val, y_test, y_val


def lemmatize_text(X_train, X_test, X_val, num_row=1):
    m = Mystem()
    def lemmatize_row(row):
        return ''.join(m.lemmatize(row))

    X_train = X_train.apply(lemmatize_row)
    X_test = X_test.apply(lemmatize_row)
    X_val = X_val.apply(lemmatize_row)
    print(X_train.head(num_row))
    return X_train, X_test, X_val


def stopwords_text(X_train, X_test, X_val, language = 'english', num_row=1):
    stop_words = set(stopwords.words(language))

    def stop_words_def(row):
        list_words = row.split()
        filtered_word = [word for word in list_words if word not in stop_words]
        return ' '.join(filtered_word)

    X_train = X_train.apply(stop_words_def)
    X_test = X_test.apply(stop_words_def)
    X_val = X_val.apply(stop_words_def)

    print(X_train.head(num_row))
    return X_train, X_test, X_val


def tf_idf_vec(X_train, X_test, X_val):

    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    X_val_tfidf = vectorizer.transform(X_val)
    return X_train_tfidf, X_test_tfidf, X_val_tfidf, vectorizer



  

max_f1_test = [0,0]
max_f1_valid = [0,0]
max_acc_test = [0,0]
max_acc_valid = [0,0]


def learning_model(data_features_train, data_target_train,  #функция принимает данные и модель
                      data_features_valid, data_target_valid,
                      data_features_test, data_target_test,
                      model, C):


  model.fit(data_features_train, data_target_train)     #обучим модель
  predicted_valid = model.predict(data_features_valid) #сделаем предсказания на валидационной выборке
  predicted_test = model.predict(data_features_test)     #сделаем предсказания на тестовой выборке
  result_valid = f1_score(data_target_valid, predicted_valid, average='weighted') #посчитаем метрику на валидационной выборке
  result_test = f1_score(data_target_test, predicted_test, average='weighted') #посчитаем метрику на тестовой выборке
  result_valid_acc = accuracy_score(data_target_valid, predicted_valid) #посчитаем метрику accuracy на валидационой выборке
  result_test_acc = accuracy_score(data_target_test, predicted_test) #посчитаем метрику accuracy на тестовой выборке

  if result_valid > max_f1_valid[1]:
    max_f1_valid[1] = result_valid
    max_f1_valid[0] = C
  if result_test > max_f1_test[1]:
    max_f1_test[1] = result_test
    max_f1_test[0] = C
  if result_valid_acc > max_acc_valid[1]:
    max_acc_valid[1] = result_valid_acc
    max_acc_valid[0] = C
  if result_test_acc > max_acc_test[1]:
    max_acc_test[1] = result_test_acc
    max_acc_test[0] = C


  return result_valid, result_test, result_valid_acc, result_test_acc


def print_score(max_f1_test, max_f1_valid, max_acc_test, max_acc_valid):
  print(f'Максимальный f1_score = {max_f1_test[1]}, полученный при значении С = {max_f1_test[0]} на тестовой выборке')
  print(f'Максимальный f1_score = {max_f1_valid[1]}, полученный при значении С = {max_f1_valid[0]} на валидационной выборке')
  print(f'Максимальный accuracy = {max_acc_test[1]}, полученный при значении С = {max_acc_test[0]} на тестовой выборке')
  print(f'Максимальный accuracy = {max_acc_valid[1]}, полученный при значении С = {max_acc_valid[0]} на валидационной выборке')


def vanga_answers(vanga):
    emotions_labels = {0: 'sadness',
                        1: 'joy',
                        2: 'love',
                        3: 'anger',
                        4: 'fear',
                        5: 'surprise'}

    for key, value in emotions_labels.items():
        if vanga[0] == key:
            print(f'You are {value}, be happy!')
