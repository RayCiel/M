import sys
import os
import jieba
sys.setdefaultencoding('utf-8')

jieba.enable_parallel(32)

for path1, path2, path3 in os.walk("E:/~RayCiel/Study/MachineLearning/TextClassification")
    for i in path3:
        input_path = path1 + "/" + i
        output_path = "E:/~RayCiel/Study/MachineLearning/TextClassification" + i
        ifstream = open(input_path, "r", "utf-8")
        ofstream = open(output_path, "w", "utf-8")
        tmp = ifstream.read(100000000)
        cut = jieba.cut(tmp)
        for j in cut:
            if j == "\n" or j == " ":
                continue
            ofstream.write(j + "\n")
