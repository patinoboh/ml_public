#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
from typing import Optional

import numpy as np
import numpy.typing as npt

import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.neural_network
import sklearn.pipeline
import sklearn.svm

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--c_n", default=5, type=int, help="Character n-grams")
parser.add_argument("--c_l", default=False, action="store_true", help="Character lowercase")
parser.add_argument("--c_tf", default="binary", type=str, help="Character TF type")
parser.add_argument("--c_mf", default=None, type=int, help="Character max features")
parser.add_argument("--c_wb", default=False, action="store_true", help="Character wb")
parser.add_argument("--cv", default=0, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--dev", default=None, type=float, help="Use given fraction as dev")
parser.add_argument("--model", default="lsvm", type=str, help="Model type")
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")
parser.add_argument("--w_n", default=2, type=int, help="Word n-grams")
parser.add_argument("--w_l", default=False, action="store_true", help="Word lowercase")
parser.add_argument("--w_tf", default="log", type=str, help="Word TF type")
parser.add_argument("--w_mf", default=None, type=int, help="Word max features")


class Dataset:
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name="nli_dataset.train.txt"):
        if not os.path.exists(name):
            raise RuntimeError("The {} was not found, please download it from ReCodEx".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        if args.dev:
            # Use the last args.dev fraction as development data.
            # When args.dev==0.1 is used, the official train/dev split is obtained.
            dev_index = int(len(train.data) * (1 - args.dev))

            train.data, train.target = train.data[:dev_index], train.target[:dev_index]
            dev = Dataset()
            dev.data, dev.target = dev.data[dev_index:], dev.target[dev_index:]

        model = sklearn.pipeline.Pipeline([
            ("feature_extraction",
             sklearn.pipeline.FeatureUnion(([
                 ("word_level", sklearn.feature_extraction.text.TfidfVectorizer(
                     lowercase=args.w_l, analyzer="word", ngram_range=(1, args.w_n),
                     binary=args.w_tf == "binary", sublinear_tf=args.w_tf == "log", max_features=args.w_mf)),
             ] if args.w_n else []) + ([
                 ("char_level", sklearn.feature_extraction.text.TfidfVectorizer(
                     lowercase=args.c_l, analyzer="char_wb" if args.c_wb else "char", ngram_range=(1, args.c_n),
                     binary=args.c_tf == "binary", sublinear_tf=args.c_tf == "log", max_features=args.c_mf)),
             ] if args.c_n else []))),
            ("estimator", {
                "lr": sklearn.linear_model.LogisticRegression(solver="saga", verbose=1),
                "mlp": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=150, max_iter=20, verbose=1),
                "svm": sklearn.svm.SVC(verbose=1),
                "lsvm": sklearn.svm.LinearSVC(),
            }[args.model]),
        ])

        if args.cv:
            scores = sklearn.model_selection.cross_val_score(model, train.data, train.target, scoring="accuracy", cv=args.cv, n_jobs=args.cv)
            print("Cross-validation with {} folds: {:.2f} +-{:.2f}".format(args.cv, 100 * scores.mean(), 100 * scores.std()))

        model.fit(train.data, train.target)
        for name, transformer in model["feature_extraction"].transformer_list:
            transformer.stop_words_ = None

        if args.dev:
            print(model.score(dev.data, dev.target))

        if args.model == "lsvm":
            model["estimator"].coef_ = model["estimator"].coef_.astype(np.float16)
        if args.model == "mlp":
            mlp = model["estimator"]
            mlp._optimizer = None
            for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
            for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)


        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
