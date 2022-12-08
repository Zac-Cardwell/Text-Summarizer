
import pandas as pd
import textSummarization as ts
#from sklearn.model_selection import train_test_split
import pickle
import numpy as np
pd.set_option("display.max_colwidth", 200)


data = pd.read_csv('Reviews.csv')
data.drop_duplicates(subset=['Text'], inplace=True)
data.dropna(axis=0, inplace=True)
Newdata = pd.read_csv('updatedReviews.csv')


def first_run(data):
    #print(data['Summary'].iloc[:5])
    cleanSumm = []
    for i in data['Summary']:
        cleanSumm.append(ts.pre_process(i, review=False))
    #print(cleanSumm[:5])

    #print(data['Text'].iloc[:5])
    cleanReview = []
    for i in data['Text']:
        cleanReview.append(ts.pre_process(i))
    #print(cleanReview[:5])

    data['cleanSumm'] = cleanSumm
    data['cleanSumm'] = data['cleanSumm'].apply(lambda x: '_START_ ' + x + ' _END_')
    data['cleanText'] = cleanReview
    data.to_csv('updatedReviews.csv', index = False)


def saveWordembed(data):

    xTrain,xTest,yTrain,yTest = train_test_split(data['cleanText'], data['cleanSumm'], test_size=0.1,random_state=0,shuffle=True)
    XtrainSeq, XtestSeq, Xvocab, YtrainSeq, YtestSeq, Yvocab = ts.wordEmbed(xTrain,xTest,yTrain,yTest)

    savedList = [XtrainSeq, XtestSeq, Xvocab, YtrainSeq, YtestSeq, Yvocab]
    with open('wordembed.txt', 'wb') as fh:
        pickle.dump(savedList, fh)


def getEmbedPickle():
    pickle_off = open('wordembed.txt', "rb")
    emp = pickle.load(pickle_off)
    XtrainSeq, XtestSeq, Xvocab, YtrainSeq, YtestSeq, Yvocab = emp[0], emp[1], emp[2], emp[3], emp[4], emp[5]
    return XtrainSeq, XtestSeq, Xvocab, YtrainSeq, YtestSeq, Yvocab


XtrainSeq, XtestSeq, Xvocab, YtrainSeq, YtestSeq, Yvocab = getEmbedPickle()




ts.createModel(XtrainSeq, YtrainSeq, Xvocab, Yvocab)






