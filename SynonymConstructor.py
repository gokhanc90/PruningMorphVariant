from scipy import spatial
import pandas as pd
import numpy as np


def getSynonymList(Synonyms, word):
    for i in range(len(Synonyms)):
        for j in range(len(Synonyms[i])):
            if word == Synonyms[i][j]:
                return Synonyms[i]
    return []


STEMMER = "SnowballEng"
COLLECTION = "CW09B"
SHEETNAME = STEMMER
BERT = 'bert-base-uncased'
COLUMN = BERT
df = pd.read_excel("npmi-" + COLLECTION + "-Bert.xlsx", sheet_name=SHEETNAME, header=0, usecols="A:E");
df = df.drop_duplicates(subset=['Term', 'Morph'], keep='last')
LOWERBOUND=df[COLUMN].quantile(0.25)-1.5*(df[COLUMN].quantile(0.75)-df[COLUMN].quantile(0.25))

# SCALE BERT SCORE
df = df[df[COLUMN] > LOWERBOUND]

for index, row in df.iterrows():
    X = row[COLUMN]
    X_std = (X - LOWERBOUND) / (1 - LOWERBOUND)

    df.at[index, COLUMN] = X_std

alpha = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
for a in alpha:
    df['Alpha' + str(a)] = np.NaN
    for index, row in df.iterrows():
        bert = row[COLUMN]
        npmi = row['npmi']

        df.at[index, 'Alpha' + str(a)] = (a * npmi + (1 - a) * bert)


# SCALE BERT SCORE END

file1 = open('Synonyms/' + COLLECTION + '/Synonym' + STEMMER + '.txt', 'r')
Lines = file1.readlines()

threshold = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
alpha = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
for a in alpha:
    for thr in threshold:
        TermList = set()

        for index, row in df.iterrows():
            score = row['Alpha' + str(a)]
            type(score)
            if score > thr:
                t = row['Term']
                m = row['Morph']
                TermList.add(t)
                TermList.add(m)

        NewLines = []
        file2 = open('Synonyms/' + COLLECTION + '/Synonym' + STEMMER + '_' + BERT + '_alpha' + str(a) + '_threshold' + str(thr) + '.txt', 'w')
        for line in Lines:
            l = []
            synonyms = line.strip().split(",")
            for w in synonyms:
                wStripped = w.strip()
                if w in TermList:
                    l.append(wStripped)
            strLine = ','.join(l)
            if len(l) > 1:
                file2.write(strLine + "\n")
        #         print(strLine)
        file2.close()