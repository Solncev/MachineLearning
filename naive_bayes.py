import csv

import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch


def main():
    test, train = get_data()
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])
    text_clf.fit(train.data, train.target)

    docs_test = test.data
    predicted = text_clf.predict(docs_test)
    print(np.mean(predicted == test.target))


def get_data(file='bayes_data/spam.csv'):
    with open(file, encoding='utf-8') as f:
        docs = csv.reader(f, delimiter=',', quotechar='"')
        data = []
        target = []
        for doc in docs:
            data.append(doc[1])
            target.append(doc[0])
        return Bunch(data=data[:len(data)//5], target=target[:len(data)//5]), \
               Bunch(data=data[len(data)//5:], target=target[len(data)//5:])

def predict(list):
    clf = joblib.load('clf.pkl')
    print(clf.predict(list))


if __name__ == '__main__':
    # main()
    predict(['Bob, buy milk on your way home.', "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"])