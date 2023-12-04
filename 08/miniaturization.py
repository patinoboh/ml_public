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
import sklearn.neural_network
import sklearn.metrics
import sklearn.pipeline
import sklearn.ensemble
import scipy
import multiprocessing
import sklearn.tree

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="miniaturization.model", type=str, help="Model path")
parser.add_argument("--stary_model", default="MNIST/mnist_competition.model", type=str, help="Model path")

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

def augment(x):
    x = x.reshape(28, 28)
    x = scipy.ndimage.zoom(x.reshape(28, 28), (np.random.uniform(0.86, 1.2), np.random.uniform(0.86, 1.2)))
    x = np.pad(x, [(2, 2), (2, 2)])
    os = [np.random.randint(size - 28 + 1) for size in x.shape]
    x = x[os[0]:os[0] + 28, os[1]:os[1] + 28]
    x = scipy.ndimage.rotate(x, np.random.uniform(-15, 15), reshape=False)
    x = np.clip(x, 0, 1)
    return x.reshape(-1)

# The following class modifies `MLPClassifier` to support full categorical distributions
# on input, i.e., each label should be a distribution over the predicted classes.
# During prediction, the most likely class is returned, but similarly to `MLPClassifier`,
# the `predict_proba` method returns the full distribution.
# Note that because we overwrite a private method, it is guaranteed to work only with
# scikit-learn 1.3.0, but it will most likely work with any 1.3.*.
class MLPFullDistributionClassifier(sklearn.neural_network.MLPClassifier):
    class FullDistributionLabels:
        y_type_ = "multiclass"

        def fit(self, y):
            return self

        def transform(self, y):
            return y

        def inverse_transform(self, y):
            return np.argmax(y, axis=-1)

    def _validate_input(self, X, y, incremental, reset):
        X, y = self._validate_data(X, y, multi_output=True, dtype=(np.float64, np.float32), reset=reset)
        if (not hasattr(self, "classes_")) or (not self.warm_start and not incremental):
            self._label_binarizer = self.FullDistributionLabels()
            self.classes_ = y.shape[1]
        return X, y
    
def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        
        with lzma.open(args.stary_model, "rb") as model_file:
            model = pickle.load(model_file)
        
        targets = model.predict_proba(train.data)
        
        #algo = sklearn.neural_network.MLPClassifier(verbose=100,hidden_layer_sizes=(682), max_iter = 100,  alpha = 0, tol =0)
        #algo = sklearn.tree.DecisionTreeClassifier(max_depth=10)
        #model = sklearn.pipeline.Pipeline([
         #   ("scaler", sklearn.preprocessing.MinMaxScaler()),
          #  ("algo", algo),
        #])
        
        model = sklearn.pipeline.Pipeline([
            ("scaler", sklearn.preprocessing.MinMaxScaler()),
            ('algos', sklearn.ensemble.VotingClassifier([
                    ("{}".format(i), sklearn.neural_network.MLPClassifier(verbose=100,hidden_layer_sizes=(650), max_iter = 100,  alpha = 0, tol =0))
                    for i in range(5)
            ], voting="soft")),
        ])
        
        data_augment = np.array(list(map(augment, train_data)))

        train_data = np.concatenate((train_data, data_augment))
        train_target = np.concatenate((train_target, data_augment))
        
        model.fit(train.data, targets)

        # TODO: Train a model on the given dataset and store it in `model`.
        
        train_predictions = model.predict(train.data)
        #result_vector = [row.index(1) for row in train_predictions]
        result_vector = np.argmax(train_predictions, axis=1)
        train_accuracy = sklearn.metrics.accuracy_score(train.target, result_vector)
        print(f"Accuracy on the training set: {train_accuracy:.2%}")
 
        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLP is in the `mlp` variable.
        
        # TODO compress
        #model = model.named_steps["algos"]
        for model in model["algos"].estimators_:
            model._optimizer = None
            for i in range(len(model.coefs_)): model.coefs_[i] = model.coefs_[i].astype(np.float16)
            for i in range(len(model.intercepts_)): model.intercepts_[i] = model.intercepts_[i].astype(np.float16)


        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        preds = model.predict(test.data)
        predictions = np.argmax(preds, axis=1)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)


# Rasto Nowak

# 6a81285c-247a-11ec-986f-f39926f24a9c

# Patrik Brocek

# 5ccdc432-238f-11ec-986f-f39926f24a9c

# Martin Oravec

# 1056cfa0-24fb-11ec-986f-f39926f24a9c
