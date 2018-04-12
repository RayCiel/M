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

#in_stopword_path = "stop_words_ch.txt"

def createjson(input, output):
    fin = open(input, "r")
    fout = open(output, "w")
    for i in range(100):
        fout.write(fin.readline())
    fout.close()

def createcsv(input, output):
    fin = open(input, "r")
    fout = open(output, "w")
    for i in range(101):
        fout.write(fin.readline())
    fout.close()



def readtrain(in_content_path, in_opinion_path):
    with open(in_content_path, "r") as in_json:
        content_prepared = []
        id_prepared = []
        ce = in_json.readlines()
        #n = 10
        for i in range(len(ce)):
            tem = json.loads(ce[i])
            content_prepared.append(tem["content"])
            id_prepared.append(tem["id"])
    with open(in_opinion_path, 'r') as in_csv:
        tmp = csv.reader(in_csv)
        column1 = [row for row in tmp]
    be = [i[1] for i in column1]
    opinion_prepared = []
    for i in range(len(be)):
        if i == 0:
            continue
        opinion_prepared.append(be[i])
    train = [id_prepared, content_prepared, opinion_prepared]
    #for i in range(10)
    # print(train)
    return train

def readtest(in_content_path):
    with open(in_content_path, "r") as in_json:
        content_prepared = []
        id_prepared = []
        ce = in_json.readlines()
        #n = 10
        for i in range(len(ce)):
            tem = json.loads(ce[i])
            content_prepared.append(tem["content"])
            id_prepared.append(tem["id"])
    test = [id_prepared, content_prepared]
    #for i in range(10)
    #print(train)
    return test

'''def stopword(in_stopword_path):
    with open(in_stopword_path, "r") as in_stopword:
        l = in_stopword.read().splitlines()
    return l'''

'''def segmentWord(input_file):
    with open(input_file, "r") as fin:
        #stopwords_list = stopword(in_stopword_path)
        c = []
        tem = fin.readlines()
        for i in range(len(tem)):
            text = ""
            tmp = tem[i]
        #    word_list = list(jieba.cut(i, cut_all=False))
            #for word in tmp:
            #    if word not in stopwords_list and word != '\r\n':
            #        text += word
            #        text += ' '
            c.append(tmp)
        return c'''



train_json_path = "train.json"
train_csv_path = "train.csv"
train10_json_path = "train10.json"
train10_csv_path = "train10.csv"
test_json_path = "test.json"
test10_json_path = "test10.json"
train_cut_path = "train_cut"
test_cut_path = "test_cut"
#createjson(train_json_path, train10_json_path)
#createcsv(train_csv_path, train10_csv_path)
#createjson(test_json_path, test10_json_path)
train = readtrain(train_json_path, train_csv_path)
#train_content = segmentWord(train_cut_path)
test = readtest(test_json_path)
#stpwrdlst = stopword(in_stopword_path)
#test_content = segmentWord(test_cut_path)


#vectorizer = CountVectorizer() #将文本中的词语转换为词频矩阵
#tfidftransformer = TfidfTransformer()
print("read done...")
n0 = len(train[1])
n1 = len(test[1])


train_opinion = [float(i[0]) for i in train[2]]

print("%d   %d" %(n0, n1))
del train
print("n1/n0 done...")
#tmp = vectorizer.fit_transform(train_content+test_content)
# print(tmp[0])
#tfidf = tfidftransformer.fit_transform(tmp)
#test_tfidf = tfidftransformer.transform(vectorizer.transform(test_content))
#test_weight = test_tfidf.toarray()
#print(test_weight)
with open('tfidf.dat', 'rb') as f:
    tfidf = pickle.loads(f.read())
print(tfidf[0])
print(train_opinion[:120])
dtrain = xgb.DMatrix(tfidf[:n0], label=train_opinion)
print("dtrain done...")
dtest = xgb.DMatrix(tfidf[n0:n0+n1])
print("dtest done...")
print(n0, n1)
print(tfidf[:n0].shape, tfidf[n0:n0+n1].shape)
print(tfidf[0],"\n############\n")

print(tfidf[1],"\n############\n")
print(tfidf[2],"\n############\n")
#param = {'max_depth':6, 'eta':0.005, 'eval_metric':'auc', 'silent':0, 'objective':'binary:logistic', 'nthread':7}
params={
'booster':'gbtree',
'objective': "binary:logistic", #多分类的问题
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':12, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.9, # 随机采样训练样本
'colsample_bytree':0.9, # 生成树时进行的列采样
'min_child_weight':3,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.01, # 如同学习率
'seed':1000,
'nthread':7,# cpu 线程数
'eval_metric':'auc'
#'eval_metric': 'auc'
}
plst = list(params.items())
evallist  = [(dtrain,'train')]
num_round = 3000
model_saved = "12-3-3000"
boost = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=100)#,  xgb_model = model_saved)
boost.save_model("12-3-3000")
print("boost done...")
preds = boost.predict(dtest)
print(preds.shape)
#print(type(preds))
with open('output-final2.csv', 'w') as f:
    f.write("id" + "," + "pred" + '\n')
    for i, pre in enumerate(preds):
        f.write(test[0][i])
        f.write(',')
        f.write(str(float(pre)))
        f.write('\n')
