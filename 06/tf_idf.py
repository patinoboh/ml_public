#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import sys
import urllib.request

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

import re

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=45, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names


def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a feature for every term that is present at least twice
    # in the training data. A term is every maximal sequence of at least 1 word character,
    # where a word character corresponds to a regular expression `\w`.    
    train_terms = [list(re.findall(r'\w+', text)) for text in train_data]
    test_terms = [list(re.findall(r'\w+', text)) for text in test_data]
    
    term_counts = {}
    for doc in train_terms:
        for term in doc:
            if term in term_counts:
                term_counts[term] += 1
            else:
                term_counts[term] = 1            

    terms = {}
    i = 0
    for key, value in term_counts.items():
        if(value >= 2):
            terms[key] = i
            i += 1
    
    print("Number of unique terms with at least two occurrences: {}".format(len(terms)))


    # TODO: For each document, compute its features as
    # - term frequency (TF), if `args.tf` is set (term frequency is
    #   proportional to counts but normalized to sum to 1);
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    
    train_features = np.zeros((len(train_data), len(terms)))
    for i, doc in enumerate(train_terms):
        for term in doc:
            if term in terms:
                j = terms[term]
                train_features[i,j] = train_features[i,j] + 1 if args.tf else 1
        if args.tf:
            train_features[i,:] /= np.sum(train_features[i,:])

    test_features = np.zeros((len(test_terms),len(terms)))
    for i, doc in enumerate(test_terms):
        for term in doc:
            if term in terms:
                j = terms[term]
                test_features[i,j] = test_features[i,j] + 1 if args.tf else 1
        if args.tf:
            test_features[i,:] /= np.sum(test_features[i,:])

    
    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.
    
    if(args.idf):
        idfs = np.zeros(len(terms))
        for i,term in enumerate(terms):
            denominator = np.sum(train_features[:,i] > 0) + 1
            idfs[i] = np.log(len(train_data)/denominator)
        train_features = train_features * idfs
        test_features = test_features * idfs

    # for dato in train_features:
    #     dato /= np.sum(dato)
    # for dato in test_features:
    #     dato /= np.sum(dato)

    # TODO: Train a `sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)`
    # model on the train set, and classify the test set.    
    
    model = sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000).fit(train_features, train_target)
    predictions = model.predict(test_features)
    
    # TODO: Evaluate the test set performance using a macro-averaged F1 score.
    f1_score = sklearn.metrics.f1_score(test_target, predictions, average="macro")

    return 100 * f1_score


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(args.tf, args.idf, f1_score))

# Rasto Nowak

# 6a81285c-247a-11ec-986f-f39926f24a9c

# Patrik Brocek

# 5ccdc432-238f-11ec-986f-f39926f24a9c

# Martin Oravec

# 1056cfa0-24fb-11ec-986f-f39926f24a9c
