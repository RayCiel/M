import sys
import os
import re
import jieba
import codecs
import json
import csv
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle as pickle

input_csv1 = "ave-90.559.csv"
#input_csv2 = "9-0.007-6000-90.126.csv"
fin_csv1 = open(input_csv1, "r")
#fin_csv2 = open(input_csv2, "r")
fout_csv = open("output-test1.csv", "w")
csv1 = csv.reader(fin_csv1)
#csv2 = csv.reader(fin_csv2)
tmp1 = []
#tmp2 = []
for line in csv1:
    #fout_csv.write(line[0] + "," + (str((float(line[1]) + float(csv2[i][1]))/2)) + '\n')
    tmp1.append(line)
#for line in csv2:
#    tmp2.append(line)
fout_csv.write("id" + "," + "pred" + '\n')
for i in range(1, len(tmp1)):
#    fout_csv.write(tmp1[i][0] + "," + (str((float(tmp1[i][1])*0.5 + float(tmp2[i][1])*0.5))) + '\n')
    if float(tmp1[i][1]) > 0.99999995:
        fout_csv.write(tmp1[i][0] + "," + str(1) + '\n')
    else:
        if float(tmp1[i][1]) < 0.00000005:
            fout_csv.write(tmp1[i][0] + "," + str(0) + '\n')
        else:
            fout_csv.write(tmp1[i][0] + "," + tmp1[i][1] + '\n')

fout_csv.close()
