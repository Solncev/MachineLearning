import math

import numpy as np

from sklearn.externals import joblib

def main():
    sample = np.empty((0, 2))

    with open('bayes_data/m.txt', 'r', encoding='utf-8') as file1:
        for line in file1:
            sample = np.append(sample, [[line.split()[0], line.split()[1]]], axis=0)

    with open('bayes_data/f.txt', 'r', encoding='utf-8') as file1:
        for line in file1:
            sample = np.append(sample, [[line.split()[0], line.split()[1]]], axis=0)

    data = np.random.permutation(sample)

    features = [(get_features(sample), int(label)) for sample, label in data]

    classifier = fit(features)
    # print(classify(classifier, get_features('Богдан')))


def fit(data):
    classes, freq = {}, {}

    for feats, label in data:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1
        for feat in feats:
            if (label, feat) not in freq:
                freq[(label, feat)] = 0
            freq[(label, feat)] += 1

    for label, feat in freq:
        freq[(label, feat)] /= classes[label]
    for c in classes:
        classes[c] /= len(data)

    return classes, freq


def classify(classifier, features):
    classes, freq = classifier
    return min(classes.keys(),
               key=lambda cl: -math.log(classes[cl]) - \
                              sum(math.log(freq.get((cl, feat), 10 ** (-7))) for feat in features))


def get_features(sample):
    return [(sample[0] + sample[-2] + sample[-1]).lower()]


def spam():
    spam = ['Путёвки по низкой цене!', 'Акция! Купи шоколадку и получи телефон в подарок']
    imp = ["Завтра состоится собрание", 'Купи килограмм яблок и шоколадку']
    test = ['В магазине гора яблок. Купи 7 килограмм и шоколадку']

    noise = ['!', '?', '.', ',']

    data = []
    for item in make_dict(spam, noise):
        data.append(([item], 0))
    for item in make_dict(imp, noise):
        data.append(([item], 1))

    classifier = fit_spam(data, 0)

    with open('bayes_data/dicts.txt', ''):
        joblib.dump(classifier, 'bayes_data/dicts.txt')

    print(classify(classifier, make_dict(["В магазине гора яблок по низкой цене, акция. Купи 7 килограмм и шоколадку"], noise)))

def make_dict(list1, noise):
    result = []
    for sent in list1:
        words = sent.split()
        # print(words)
        for word in words:
            for n in noise:
                word = word.replace(n, '')
            if len(word) > 3:
                result += [word.lower()]
    return result


def fit_spam(data, alpha):
    classes, freq, tot = {}, {}, set()

    for feats, label in data:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1
        for feat in feats:
            if (label, feat) not in freq:
                freq[(label, feat)] = 0
            freq[(label, feat)] += 1
        tot.add(tuple(feats))

    for label, feat in freq:
        freq[(label, feat)] = (alpha + freq[(label, feat)]) / (alpha * len(tot) + classes[label])
    for c in classes:
        classes[c] /= len(data)

    return classes, freq

if __name__ == '__main__':
    # main()
    spam()
