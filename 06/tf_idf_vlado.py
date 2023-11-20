#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import re
import sys
import urllib.request

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

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
    train_docs = [re.findall(r'\w+', doc) for doc in train_data]
    test_docs = [re.findall(r'\w+', doc) for doc in test_data]
    

    terms = {}
    for doc in train_docs:
        for term in doc:
            if term not in terms:
                terms[term] = 1
            else:
                terms[term] += 1
                
    to_del = []
    for key in terms:
        if terms[key] < 2:
            to_del.append(key)
    for key in to_del:
        del terms[key]

    term_count = 0
    for term in terms:
        terms[term] = term_count
        term_count += 1


    # TODO: For each document, compute its features as
    # - term frequency (TF), if `args.tf` is set (term frequency is
    #   proportional to counts but normalized to sum to 1);
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    #
    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.
    
    idf_mask = np.ones(len(terms))
    if args.idf:
        in_docs = np.zeros(len(terms))
        for doc in train_docs:
            for term in set(doc):
                if term in terms:
                    in_docs[terms[term]] += 1
                    
        idf_mask = np.log(len(train_data)/(in_docs + 1))

    train_vectors = []
    test_vectors = []
    for vectors, docs in [(train_vectors, train_docs), (test_vectors, test_docs)]:
        for doc in docs:
            vec = np.zeros(len(terms))     

            for term in doc:
                if term in terms:
                    if args.tf:
                        vec[terms[term]] += 1
                    else:
                        vec[terms[term]] = 1
            if args.tf:
                vec /= np.sum(vec)
            vec *= idf_mask
            vectors.append(vec)
                

    # TODO: Train a `sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)`
    # model on the train set, and classify the test set.
    est = sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000).fit(train_vectors, train_target)
    # TODO: Evaluate the test set performance using a macro-averaged F1 score.
    test_pred = est.predict(test_vectors)
    f1_score = sklearn.metrics.f1_score(test_target, test_pred, average="macro")

    return 100 * f1_score


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(args.tf, args.idf, f1_score))