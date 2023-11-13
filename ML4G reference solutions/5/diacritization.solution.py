#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.pipeline

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--dictionary", default=False, action="store_true", help="Use the dictionary")
parser.add_argument("--dev", default=None, type=float, help="Use given fraction as dev")
parser.add_argument("--hidden_layers", nargs="+", default=[100], type=int, help="Hidden layer sizes")
parser.add_argument("--max_iter", default=100, type=int, help="Max iters")
parser.add_argument("--model", default="lr", type=str, help="Model to use")
parser.add_argument("--model_kind", default="single", type=str, help="Model kind (single/per_letter)")
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--prune", default=0, type=int, help="Prune features with <= given counts")
parser.add_argument("--solver", default="saga", type=str, help="LR solver")
parser.add_argument("--target_mode", default="marks", type=str, help="Target mode (letters/marks)")
parser.add_argument("--window_chars", default=1, type=int, help="Window characters to use")
parser.add_argument("--window_ngrams", default=4, type=int, help="Window ngrams to use")


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

    TARGET_LETTERS = sorted(set(LETTERS_NODIA + LETTERS_DIA))
    @staticmethod
    def letter_to_target(letter, target_mode):
        if target_mode == "letters":
            return Dataset.TARGET_LETTERS.index(letter)
        elif target_mode == "marks":
            if letter in "áéíóúý":
                return 1
            if letter in "čďěňřšťůž":
                return 2
            return 0

    @staticmethod
    def target_to_letter(target, letter, target_mode):
        if target_mode == "letters":
            return Dataset.TARGET_LETTERS[target]
        elif target_mode == "marks":
            if target == 1:
                index = "aeiouy".find(letter)
                return "áéíóúý"[index] if index >= 0 else letter
            if target == 2:
                index = "cdenrstuz".find(letter)
                return "čďěňřšťůž"[index] if index >= 0 else letter
            return letter

    def get_features(self, args):
        processed = self.data.lower()
        features, targets, indices = [], [], []
        for i in range(len(processed)):
            if processed[i] not in Dataset.LETTERS_NODIA:
                continue
            features.append([processed[i]])
            for o in range(1, args.window_chars):
                features[-1].append(processed[i - o:i - o + 1])
                features[-1].append(processed[i + o:i + o + 1])
            for s in range(1, args.window_ngrams):
                for o in range(-s, 0+1):
                    features[-1].append(processed[max(i + o, 0):i + o + s + 1])
            targets.append(self.letter_to_target(self.target[i].lower(), args.target_mode))
            indices.append(i)

        return features, targets, indices


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        train_data, train_target, _ = train.get_features(args)
        if args.prune:
            for i in range(len(train_data[0])):
                features = {}
                for data in train_data:
                    features[data[i]] = features.get(data[i], 0) + 1
                for data in train_data:
                    if features[data[i]] <= args.prune:
                        data[i] = "<unk>"

        def create_model():
            return sklearn.pipeline.Pipeline([
                ("one-hot", sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")),
                ("estimator", {
                    "lr": sklearn.linear_model.LogisticRegression(solver=args.solver, multi_class="multinomial", max_iter=args.max_iter, verbose=1),
                    "lrr": sklearn.linear_model.LogisticRegressionCV(solver=args.solver, Cs=np.geomspace(1, 1e4, 10), max_iter=args.max_iter, cv=10, n_jobs=-1),
                    "mlp": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=args.hidden_layers, max_iter=args.max_iter, verbose=1),
                }[args.model]),
            ])

        def postprocess_model(model):
            if args.model == "mlp":
                mlp = model["estimator"]
                mlp._optimizer = None
                for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
                for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)
            if args.model == "lrr":
                model["estimator"].coefs_paths_ = None
                print("Finished training, chosen C {}".format(model["estimator"].C_), file=sys.stderr)

        if args.model_kind == "single":
            if args.dev:
                train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
                    train_data, train_target, test_size=args.dev, shuffle=False)

            model = create_model()
            model.fit(train_data, train_target)
            postprocess_model(model)

            if args.dev:
                print("Development accuracy: {}%".format(100 * model.score(test_data, test_target)))
        else:
            assert args.target_mode == "marks", "The per-letter models support only `marks` target."
            assert args.dev is None, "The per-letter models do not support `dev` argument."

            model = {}
            for i in range(len(train_data)):
                letter = train_data[i][0]
                if letter in model:
                    continue
                print("Training letter {}".format(letter), file=sys.stderr)

                model[letter] = create_model()
                model[letter].fit([x for x in train_data if x[0] == letter], [t for x, t in zip(train_data, train_target) if x[0] == letter])
                postprocess_model(model[letter])

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump((model, args), model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model, model_args = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        test_data, _, test_indices = test.get_features(model_args)

        if model_args.model_kind == "single":
            test_target = model.predict(test_data)
        else:
            test_by_letter, target_by_letter, test_target = {}, {}, []
            for x in test_data:
                test_by_letter.setdefault(x[0], []).append(x)
            for letter in test_by_letter:
                target_by_letter[letter] = list(model[letter].predict(test_by_letter[letter]))
            for x in test_data:
                test_target.append(target_by_letter[x[0]].pop(0))

        predictions = list(test.data)
        for i in range(len(test_target)):
            predictions[test_indices[i]] = test.target_to_letter(test_target[i], test.data[test_indices[i]].lower(), model_args.target_mode)
            if test.data[test_indices[i]].isupper():
                predictions[test_indices[i]] = predictions[test_indices[i]].upper()
        predictions = "".join(predictions)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
