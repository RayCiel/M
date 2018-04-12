# -*- coding:utf-8 -*-
import xgboost as xgb
import csv
import jieba
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sys
import importlib
import pickle
importlib.reload(sys)


'''def segmentWord(input_file):
    with open(input_file, "r") as fin:
        #stopwords_list = stopword(in_stopword_path)
        c = []
        tem = fin.readlines()
        for i in range(len(tem)):
            #text = ""
        #    word_list = list(jieba.cut(i, cut_all=False))
            #for word in tmp:
            #    if word not in stopwords_list and word != '\r\n':
            #        text += word
            #        text += ' '
            c.append(tem[i])
        return c
'''
train_cut_path = "train_cut10"
test_cut_path = "test_cut10"
in_stopword_path = "stop_words_ch.txt"
with open(train_cut_path, "r") as fin:
    train_content = fin.read().splitlines()
print("train_contest done...")
with open(test_cut_path, "r") as fin2:
    test_content = fin2.read().splitlines()
print("test_contest done...")
with open(in_stopword_path, "r") as in_stopword:
    stpwrdlst = in_stopword.read().splitlines()
print("stopwordlist done...")
#train_content = segmentWord(train_cut_path)
#test_content = segmentWord(test_cut_path)
vectorizer = CountVectorizer(stop_words = stpwrdlst)
del stpwrdlst
tfidftransformer = TfidfTransformer()
tmp = vectorizer.fit_transform(train_content+test_content)
print("vectorizer done...")
del train_content
del test_content
tfidf = tfidftransformer.fit_transform(tmp)
print("tf done...")
with open('tfidf10.dat', 'wb') as f:
    pickle.dump(tfidf, f)
