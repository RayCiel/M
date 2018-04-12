import sys
import os
import re
import jieba
import codecs
import json
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle as pickle


def change(input_file, output_file):
    fin = open(input_file, "r")
    fout = open(output_file, "w")
    tmp = fin.readlines()
    for i in range(len(tmp)):
        if i % 100 == 0 :
            print("dealing %d..."%i)
        text_prepared = []
        tem = json.loads(tmp[i])
        ce = re.compile(u'[^\u4E00-\u9FA5]+')
        text_content = tem["content"]
        text_content = re.sub(ce, "", text_content, 0)
        cut = jieba.cut(text_content)
        fout.write(" ".join(cut) + "\n")
    fout.close()
print("Cutting train:")
input_file = "train.json"
output_file = "train_cut"
change(input_file, output_file)

'''print("Cutting test:")
input_file = "test10.json"
output_file = "test_cut10"
change(input_file, output_file)
'''
