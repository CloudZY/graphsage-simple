from __future__ import print_function
import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from time import time
from .calc_node_degree import get_node_degree, get_node_degree_dp_distribution
import math

import json


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[int(x)] for x in X]
        # X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        # averages = ["micro", "macro", "samples", "weighted"]

        # results = {}
        # for average in averages:
        # results[average] = f1_score(Y, Y_, average=average)
        # print('Results, using embeddings of dimensionality', len(self.embeddings[X[0]]))
        # print('-------------------')
        # results["macro"] = f1_score(Y, Y_, average="macro")
        # print(results)
        # return results
        # print('-------------------')
        return f1_score(Y, Y_, average="macro")

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[int(x)] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]

        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        # Get the X_test and Y_test
        # split by node degree
        # evaluate based on node degree, get f1 score

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)

        # get the node degree dict
        degree = get_node_degree()
        # dict for storing sorted degree value
        sorted_degree = {}
        degree_f1_scores = {}

        # classify them by node degree
        for (x_val, y_val) in zip(X_test, Y_test):
            # find its node degree
            deg = degree[x_val]
            if deg not in sorted_degree:
                sorted_degree[deg] = [[] for i in range(0, 2)]
                # append x, y values to sorted degree
            sorted_degree[deg][0].append(x_val)
            sorted_degree[deg][1].append(y_val)

        # when evaluate, evaluate based on their node degrees
        for deg in sorted_degree.keys():
            x_vals = sorted_degree[deg][0]
            y_vals = sorted_degree[deg][1]
            # print("Node dgree: ", deg)
            # print(sorted_degree[deg])
            # print(self.evaluate(x_vals, y_vals))
            print(sorted_degree[deg])
            degree_f1_scores[deg] = self.evaluate(x_vals, y_vals)

        with open("n2v_degree_f1.json", "w") as fi:
            json.dump(degree_f1_scores, fi)

        # print(X_test, Y_test)
        # return self.evaluate(X_test, Y_test)

    # def split_train_evaluate_based_on_degree(self, X, Y, train_percent, group_id, seed=0):
    #     state = numpy.random.get_state()
    #     degree_distribution = get_node_degree_dp_distribution()
    #     X_all = degree_distribution[group_id]
    #     training_size = int(train_percent * len(X_all))
    #     numpy.random.seed(seed)

    #     # Train & Test Based on Node Degree
    #     shuffle_indices = numpy.random.permutation(numpy.arange(len(X_all)))

    #     Y_all = []
    #     for x in X_all:
    #         ind = X.index(str(x))
    #         Y_all.append(Y[ind])

    #     X_train = [X_all[shuffle_indices[i]] for i in range(training_size)]
    #     Y_train = [Y_all[shuffle_indices[i]] for i in range(training_size)]

    #     X_test = [X_all[shuffle_indices[i]] for i in range(training_size, len(X_all))]
    #     Y_test = [Y_all[shuffle_indices[i]] for i in range(training_size, len(X_all))]

    #     self.train(X_train, Y_train, Y_all)
    #     numpy.random.set_state(state)

    #     res = self.evaluate(X_test, Y_test)
    #     # print (self.evaluate(X_test, Y_test))
    #     # print("\n")

    #     with open("degree_based_train.txt", "a") as f:
    #         f.write(str(group_id) + " " + str(res) + "\n")

    def split_train_evaluate_based_on_degree(self, X, Y, train_percent, seed=0):
        node_degree_dict = get_node_degree()
        state = numpy.random.get_state()

        degree_dict = sorted(node_degree_dict.items(), key=lambda kv: kv[1])

        X_all = []
        Y_all = []
        for x, y in degree_dict:
            if y == 10:
                X_all.append(int(x))
                ind = X.index(str(x))
                Y_all.append(Y[ind])
        print(X_all)

        training_size = int(train_percent * len(X_all))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X_all)))
        X_train = [X_all[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y_all[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X_all[shuffle_indices[i]] for i in range(training_size, len(X_all))]
        Y_test = [Y_all[shuffle_indices[i]] for i in range(training_size, len(X_all))]
        self.train(X_train, Y_train, Y_all)
        numpy.random.set_state(state)
        res = self.evaluate(X_test, Y_test)

        with open("degree_10_gs.txt", "a") as f:
            f.write(str(res) + "\n")

    def split_train_evaluate_based_on_group(self, X, Y, train_precent, num_groups, fname, seed=0):
        node_degree_dict = get_node_degree()
        state = numpy.random.get_state()
        start = 0
        end = 10312
        group_range = math.floor(end / num_groups)

        for i in range(num_groups):
            if i == num_groups - 1:
                X_all = X[start: end]
                Y_all = Y[start: end]

            else:
                X_all = X[start: start + group_range]
                Y_all = Y[start: start + group_range]

            # Get the min & max degree for each group
            min_degree = node_degree_dict[str(X[start])]
            max_degree = node_degree_dict[str(X[start + group_range])]

            training_size = int(train_precent * len(X_all))
            numpy.random.seed(seed)
            shuffle_indices = numpy.random.permutation(numpy.arange(len(X_all)))
            X_train = [X_all[shuffle_indices[i]] for i in range(training_size)]
            Y_train = [Y_all[shuffle_indices[i]] for i in range(training_size)]
            X_test = [X_all[shuffle_indices[i]] for i in range(training_size, len(X_all))]
            Y_test = [Y_all[shuffle_indices[i]] for i in range(training_size, len(X_all))]

            self.train(X_train, Y_train, Y_all)
            numpy.random.set_state(state)
            res = self.evaluate(X_test, Y_test)
            with open(fname, "a") as f:
                f.write(str(res) + " " + str(min_degree) + " " + str(max_degree) + "\n")
            start = start + group_range

            # if i >= 5:
            #     with open("partial_data.txt", "a") as f:
            #         f.write(''.join(str(x) + " " for x in X_all))

    def split_train_evaluate_graphsage(self, X, Y, train_percent, ids, seed=0):
        X_actual = ids
        Y_actual = []
        for x in ids:
            ind = X.index(str(x))
            Y_actual.append(Y[ind])

        state = numpy.random.get_state()
        training_size = int(train_percent * len(X_actual))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X_actual)))
        X_train = [X_actual[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y_actual[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X_actual[shuffle_indices[i]] for i in range(training_size, len(X_actual))]
        Y_test = [Y_actual[shuffle_indices[i]] for i in range(training_size, len(X_actual))]

        self.train(X_train, Y_train, Y_actual)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)


def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size + 1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors


def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y