from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from .classify import Classifier, read_node_label
from .file_reader import parse_vector, parse_graphsage_vector, get_graphsage_vectors, concate_embed
from .calc_node_degree import sort_data_by_degree, degree_id_dict



def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification')


def main(args):
    degree_dict = degree_id_dict()
    vectors = concate_embed(degree_dict)
    # Sorted X, Y
    X_sorted, Y_sorted = sort_data_by_degree()
    print(X_sorted)
    print(Y_sorted)
    group_size = 10
    clf = Classifier(vectors=vectors, clf=LogisticRegression())
    clf.split_train_evaluate_based_on_group(X_sorted, Y_sorted, args.clf_ratio, group_size,
                                            "group_based_train_" + str(group_size) + "gs_degree_rand.txt")


def experiment_n2v(args):
    if args.label_file and args.method != "gcn":
        vectors = parse_vector()
        X, Y = read_node_label(args.label_file)
        print("Training classifier using {:.2f}% nodes...".format(args.clf_ratio * 100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        clf.split_train_evaluate(X, Y, args.clf_ratio)


if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main(parse_args())
    # experiment_n2v(parse_args())
