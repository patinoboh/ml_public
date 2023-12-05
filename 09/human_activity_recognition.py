#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.pipeline

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="human_activity_recognition.model", type=str, help="Model path")


class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(self,
                 name="human_activity_recognition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and if it contains column "class", split it to `targets`.
        self.data = pd.read_csv(name)
        if "class" in self.data:
            self.target = np.array([Dataset.CLASSES.index(target) for target in self.data["class"]], np.int32)
            self.data = self.data.drop("class", axis=1)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        
        algo = sklearn.ensemble.HistGradientBoostingClassifier(),
        model = sklearn.pipeline.Pipeline([
            ("scaler", sklearn.preprocessing.StandardScaler()),
            ("algo", algo)])
        
        # TODO CV
        # este iba max_depth :[4,6,8,10]
        """grid = {"algo__max_iter": [500,600,700], "algo__max_leaf_nodes":[10,20,30,"None"], "algo__early_stopping":[False, True,"auto"] }
        model = sklearn.model_selection.GridSearchCV(estimator=model,  
                                                      cv = sklearn.model_selection.StratifiedKFold(5), 
                                                      param_grid=grid, 
                                                      n_jobs=1, 
                                                      refit=True,
                                                      verbose =100,
                                                      scoring='accuracy')"""
        
        scores = sklearn.model_selection.cross_val_score(model, train.data, train.target, cv=sklearn.model_selection.StratifiedKFold(5), n_jobs=7, scoring = 'accuracy')
        print(args.cv, 100 * scores.mean(), 100 * scores.std())
        
        model.fit(train.data, train.target)
        
        
        
        #print(model.best_params_)
        
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
