import csv
import json

with open("test.json", "r") as fin:
    ce = fin.readlines()
    id = []
    for i in range(len(ce)):
        tem = json.loads(ce[i])
        id.append(tem["id"])

with open("randomttest.csv", "w") as fout:
    fout.write("id" + "," + "pred" + '\n')
    for i in range(len(ce)):
        fout.write(id[i] + "," + "0.5" + '\n')
