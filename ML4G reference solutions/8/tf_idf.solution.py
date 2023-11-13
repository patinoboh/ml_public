#!/usr/bin/env python3
import argparse
import collections
import lzma
import pickle
import os
import re
import sys
import urllib.request

import numpy as np
import sklearn.feature_extraction
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.pipeline

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=177, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=1000, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--with_reference", default=False, action="store_true", help="Show also reference results")


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

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

    terms = collections.Counter(term for doc in train_terms for term in doc)
    terms = dict((term, i) for i, term in enumerate(term for term, count in terms.items() if count >= 2))
    print("Number of unique terms with at least two occurrences: {}".format(len(terms)))

    # TODO: For each document, compute its features as
    # - term frequency(TF), if `args.tf` is set;
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    #
    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.
    idf = np.ones(len(terms))
    if args.idf:
        for doc in train_terms:
            for term in set(doc) & terms.keys():
                idf[terms[term]] += 1
        idf = np.log(len(train_data) / idf)

    train_vectors, test_vectors = [], []
    for dataset, target in [(train_terms, train_vectors), (test_terms, test_vectors)]:
        for doc in dataset:
            vector = np.zeros(len(terms))
            for term in doc if args.tf else set(doc):
                if term in terms:
                    vector[terms[term]] += 1
            assert any(vector != 0), "Found a document with no features"

            vector *= idf
            vector /= np.linalg.norm(vector)
            target.append(vector)

    # TODO: Perform classification of the test set using the k-NN algorithm
    # from sklearn (pass the `algorithm="brute"` option), with `args.k` nearest
    # neighbors. For TF-IDF vectors, the cosine similarity is usually used, where
    #   cosine_similarity(x, y) = x^T y / (||x|| * ||y||).
    #
    # To employ this metric, you have several options:
    # - you could try finding out whether `KNeighborsClassifier` supports it directly;
    # - or you could compute it yourself, but if you do, you have to precompute it
    #   in a vectorized way, so using `metric="precomputed"` is fine, but passing
    #   a callable as the `metric` argument is not (it is too slow);
    # - finally, the nearest neighbors according to cosine_similarity are equivalent to
    #   the neighbors obtained by the usual Euclidean distance on L2-normalized vectors.
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=args.k, algorithm="brute")
    predictions = knn.fit(train_vectors, train_target).predict(test_vectors)

    # TODO: Evaluate the performance using a macro-averaged F1 score.
    f1_score = sklearn.metrics.f1_score(test_target, predictions, average="macro")

    if args.with_reference:
        model = sklearn.pipeline.Pipeline([
            ("tf-idf", sklearn.feature_extraction.text.TfidfVectorizer(use_idf=args.idf, binary=not args.tf, smooth_idf=False)),
            ("knn", knn),
        ])
        predictions = model.fit(train_data, train_target).predict(test_data)
        print("Reference solution: {:.1f}%".format(
            100 * sklearn.metrics.f1_score(test_target, predictions, average="macro")))

    return f1_score


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}, k={}: {:.1f}%".format(args.tf, args.idf, args.k, 100 * f1_score))
