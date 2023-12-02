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

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="miniaturization.model", type=str, help="Model path")


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
    
    
def scale(image, scale1, scale2):
    image = scipy.ndimage.zoom(image.reshape(28,28), (np.random.uniform(scale1,scale2), np.random.uniform(scale1,scale2)))
    return image

def add_padding(image, pad_size):
    image = np.pad(image, [(pad_size, pad_size), (pad_size, pad_size)])
    return image

def crop(image, crop_size):
    x = [np.random.randint(size - crop_size + 1) for size in image.shape]
    image = image[x[0]:x[0] + crop_size, x[1]:x[1] + crop_size]
    return image

def rotate(image, rotate1, rotate2):
    image = scipy.ndimage.rotate(image, np.random.uniform(rotate2, rotate2), reshape=False)
    return image

    
def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        
        
        
        #TODO teacher and student
        
        #TODO DATA AUGMENT
        augmented_data = []
        for image in train.data:
            image.reshape(28, 28)
            scaled_img = scale(image, 0.9,1.1)
            padded_img = add_padding(scaled_img, 2)
            cropped_img = crop(padded_img, 28)
            rotated_img = rotate(cropped_img, -15, 15)
            rotated_img= np.clip(rotated_img, 0, 1)
            rotated_img= rotated_img.reshape(-1)
            augmented_data.append(rotated_img)

        combined_data = np.vstack([train.data, augmented_data])
        #combined_data = np.append(train.data, np.array(augmented_data), axis=0)
        combined_targets = np.concatenate([train.target, train.target])
        
        model = sklearn.pipeline.Pipeline([
            ("scaler", sklearn.preprocessing.MinMaxScaler()),
            ('algos', sklearn.ensemble.VotingClassifier([
                    ("{}".format(i), sklearn.neural_network.MLPClassifier(verbose=100,hidden_layer_sizes=(650), max_iter = 1,  alpha = 0, tol =0))
                    for i in range(2)
            ], voting="soft")),
        ])
        
        #model.fit(train.data, train.target)
        model.fit(combined_data, combined_targets) 
        
        train_predictions = model.predict(train.data)
        train_accuracy = sklearn.metrics.accuracy_score(train.target, train_predictions)
        print(f"Accuracy on the training set: {train_accuracy:.2%}")

        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLP is in the `mlp` variable.
        
        #model = model.named_steps["algo"]
        for model in model['algos'].estimators_:
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
        predictions = model.predict(test.data)

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
