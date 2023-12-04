#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
import sklearn.model_selection
from sklearn.neural_network import MLPClassifier
import sklearn.pipeline

import scipy
from PIL import Image


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")


class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)


def augment(image):
    image = image.reshape(28,28)
    image = scipy.ndimage.rotate(image, np.random.uniform(-15,15), reshape=False)
    # image = scipy.ndimage.zoom(image, np.random.uniform(0.88, 1.2)) # I didn't have time to play with zoom properly
    image = scipy.ndimage.shift(image,(np.random.uniform(-1.2,1.2), np.random.uniform(-1.2,1.2)))
    image = image.reshape(-1)
    return image


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Train a model on the given dataset and store it in `model`.
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
            train.data, train.target, test_size=0.2, random_state=args.seed)

        # For every image I generate three similar ones
        first_group = np.array(list(map(augment, train_data)))
        second_group = np.array(list(map(augment, train_data)))
        third_group = np.array(list(map(augment, train_data)))

        train_data = np.concatenate((train_data, first_group, second_group, third_group))
        train_target = np.concatenate((train_target, train_target, train_target, train_target))


        pipe = sklearn.pipeline.Pipeline([
            ("scaler", sklearn.preprocessing.MinMaxScaler()),
            ("ensemble", sklearn.ensemble.VotingClassifier(
                [("mlp1", MLPClassifier(random_state=args.seed,
                        # max_iter=300,
                        hidden_layer_sizes=(500),
                        learning_rate="adaptive",
                        early_stopping=True,
                        verbose=True,
                        tol = 0,
                        )),
                ("mlp2", MLPClassifier(random_state=2*args.seed,
                        # max_iter=300,
                        hidden_layer_sizes=(500),
                        learning_rate="adaptive",
                        early_stopping=True,
                        verbose=True,
                        tol = 0,
                        )),
                ("mlp3", MLPClassifier(random_state=3*args.seed,
                        # max_iter=300,
                        hidden_layer_sizes=(500),
                        learning_rate="adaptive",
                        early_stopping=True,
                        verbose=True,
                        tol = 0,
                        )),
                ("mlp4", MLPClassifier(random_state=4*args.seed,
                        # max_iter=300,
                        hidden_layer_sizes=(500),
                        learning_rate="adaptive",
                        early_stopping=True,
                        verbose=True,
                        tol = 0,
                        )),
                ("mlp5", MLPClassifier(random_state=5*args.seed,
                        # max_iter=300,
                        hidden_layer_sizes=(500),
                        learning_rate="adaptive",
                        early_stopping=True,
                        verbose=True,
                        tol = 0,
                        )),
                ],
                
                verbose = True,
                voting = "soft",
                n_jobs = 6,))
        ])


        pipe.fit(train_data, train_target)

        test_predict = pipe.predict(test_data)
        train_predict = pipe.predict(train_data)

        train_accuracy = sklearn.metrics.accuracy_score(train_target, train_predict)
        test_accuracy = sklearn.metrics.accuracy_score(test_target, test_predict)

        print("train", train_accuracy)
        print("test", test_accuracy)

        model = pipe

        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained `MLPClassifier` is in the `mlp` variable.
        #   mlp._optimizer = None
        #   for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        #   for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)


        # Making smaller model, unintuitivly in prediction part of code.
        # I used it to play with this part of code without waiting for training again
        # for mlp in model["ensemble"].estimators_:
        #     mlp._optimizer = None
        #     for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        #     for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)
        #
        # with lzma.open("mnist_competition.smaller.model", "wb") as model_file:
        #     pickle.dump(model, model_file)

        # Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)
        # test_accuracy = sklearn.metrics.accuracy_score(test.target, predictions)
        # print(test_accuracy)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)