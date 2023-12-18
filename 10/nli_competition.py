#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
from typing import Optional

import numpy as np
import numpy.typing as npt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from sklearn.svm import LinearSVC

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")


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

        char_vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=.3, min_df=5, lowercase=False, analyzer='char', ngram_range=(1,6), binary=True, max_features=100000, decode_error='ignore')
        word_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=.3, min_df=5, lowercase=True, analyzer='word', ngram_range=(1,2), decode_error='ignore')
        
        X_train, X_test, y_train, y_test = train_test_split(train.data, train.target, test_size=0.1, random_state=42)

        char_features = char_vectorizer.fit_transform(X_train)
        word_features = word_vectorizer.fit_transform(X_train)
        train_targets = y_train
        
        train_features = hstack([char_features, word_features])

        # TODO: Train a model on the given dataset and store it in `model`.
        model = LinearSVC().fit(train_features, train_targets)

        test_features = hstack([char_vectorizer.transform(X_test), word_vectorizer.transform(X_test)])
        print(accuracy_score(y_test, model.predict(test_features)))
        
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump((model, char_vectorizer, word_vectorizer), model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model, char_vectorizer, word_vectorizer = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        features = hstack([char_vectorizer.transform(test.data), word_vectorizer.transform(test.data)])
        predictions = model.predict(features)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
