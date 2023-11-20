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
parser.add_argument("--tf", default=True, action="store_true", help="Use TF weights")
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

# def tf(t, d, binary = False):    
    # if binary:
    # else:




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
    
    flat_terms = [term for doc in train_terms for term in doc]
    term_counts = {term: flat_terms.count(term) for term in flat_terms}
    terms = []
    for key, value in term_counts.items():
        if(value >= 2):
            terms.append(key)

    # TODO: For each document, compute its features as
    # - term frequency (TF), if `args.tf` is set (term frequency is
    #   proportional to counts but normalized to sum to 1);
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    tfs = np.zeros((len(train_data),len(terms)))
    for i, doc in enumerate(train_terms):
        for j, term in enumerate(doc):
            if term in terms:
                tfs[i,j] = tfs[i,j] + 1 if args.tf else 1
        if(args.tf):
            tfs[i,] /= np.sum(tfs[i,])

    tfs = np.zeros((len(train_data),len(terms)))
    




    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.

    # TODO: Train a `sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)`
    # model on the train set, and classify the test set.
    
    model = sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)
    # model.fit()
    # model.predict()        

    # TODO: Evaluate the test set performance using a macro-averaged F1 score.
    f1_score = 10

    return 100 * f1_score


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(args.tf, args.idf, f1_score))