import xgboost as xgb
import csv
import jieba
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sys
import importlib
in_stopword_path = "stop_words_ch.txt"
def stopword(in_stopword_path):
    with open(in_stopword_path, "r") as in_stopword:
        l = in_stopword.read().splitlines()
    return l
stopword(in_stopword_path)
