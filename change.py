#-*- coding:unicode_escape -*-
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
#reload(sys)
#sys.setdefaultencoding('utf-8')

jieba.enable_parallel(32)
'''ifstream = open("test.json", "r")
ofstream = open("test-utf-8", "w")
#while 1:
for i in range(0, 10):
    tmp = ifstream.readlines()
#    if tmp == "":
    #    break
    n = len(tmp)
    for j in range(n):
        tmp[j] = tmp[j][:-1]
        tmp[j] = tmp[j].encode('latin-1').decode('unicode_escape')
        ofstream.write(tmp[j] + '\n')
ofstream.close()'''

input_file = "test.json"
output_file = "test-jieba"
stopword_path = "stopwords.dat"
wordbag_path = "test_word.dat"
stopword=[]


def change(input_file, output_file):
    fin = open(input_file, "r")
    fout = open(output_file, "w")
    stop_fin = open("stopwords.txt", "r")
    stop_text = stop_fin.readlines()
    n = len(stop_text)

    for i in range(n):
        stopword.append(stop_text[i][:-1])#.encode('unicode_escape').decode('utf-8')[:-1])
    bunch_obj = Bunch(id=[], content=[], label=[])
    for i in range(1):
        text_prepared = []
        tem = json.loads(fin.readline())
        ce = re.compile(u'[^\u4E00-\u9FA5]+')
        text_content = tem["content"]
        text_content = re.sub(ce, "", text_content, 0)
        cut = jieba.cut(text_content)
        fout.write("id: " + tem["id"] + "\\n")
        bunch_obj.id.append(tem["id"])
        for s in cut:



        #flag = 0
        #print(s)
            if s in stopword:
            #flag = 1
                continue
            text_prepared.append(s)
            s = s + "\\n"
            fout.write(s)
        bunch_obj.content = text_prepared.copy()
        #bunch_obj.content.append(text_prepared)
    with open(wordbag_path, "wb") as file_obj:
            pickle.dump(bunch_obj, file_obj)
    fout.close()

def _writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)

def _readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content

def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = Bunch(id = [])#pickle.load(file_obj)
    return bunch

def vector_space(stopword,bunch_path,space_path,train_tfidf_path = None):
    stpwrdlst = _readfile(stopword_path).splitlines()
    bunch = _readbunchobj(bunch_path)
    tfidfspace = Bunch(id = bunch.id.copy(), content = bunch.content.copy(), tdm = [], vocabulary = {})
    if train_tfidf_path is not None:
        trainbunch = _readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words = stpwrdlst, sublinear_tf = True, max_df = 0.5, vocabulary = trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.content)

    else:
        vectorizer = TfidfVectorizer(stop_words = stpwrdlst, sublinear_tf = True, max_df = 0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.content)
        tfidfspace.vocabulary = vectorizer.vocabulary_
    _writebunchobj(space_path, tfidfspace)


def main():
    change(input_file, output_file)
    vector_space(stopword, "train.json", "tfidf")
    trainpath = "tfdifspace.dat"
    train_set = _readbunchobj(trainpath)

    testpath = "test.json"
    test_set = _readbunchobj(testpath)

    clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)

    predicted = clf.predict(test_set.tdm)

    #for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    #    if flabel != expct_cate:
    #        print file_name,": 实际类别:",flabel," -->预测类别:",expct_cate

main()
