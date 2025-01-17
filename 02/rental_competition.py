#!/usr/bin/env python3

import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import sklearn
import sklearn.pipeline
import sklearn.compose
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection

import sklearn.dummy


import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def get_integer_columns(matrix):
    cols = []    
    for col_num, column in enumerate(matrix.transpose()):
        integers = True
        for data in column:
            integers &= data.is_integer()
        if(integers):
            cols.append(col_num)
    return np.array(cols)


def create_model(data):    
    int_columns = get_integer_columns(data.data)
    
    model = [("algo", sklearn.linear_model.PoissonRegressor(max_iter=500, alpha=1, verbose=3))]

    model = sklearn.pipeline.Pipeline(
        [("preprocess", sklearn.compose.ColumnTransformer(
            [("onehot", sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False), int_columns),
             ("scaler", sklearn.preprocessing.StandardScaler(), ~int_columns),])), 
             ("poly", sklearn.preprocessing.PolynomialFeatures(2)) ]       
        + model
        )
    
    #alphas = (0.6, 0.9, 1,)
    alphas = np.arange(0, 1, 0.1)
    max_iters = np.arange(100, 1000,100)
    
    cross_valid = sklearn.model_selection.StratifiedKFold(5)
    params = {"poly__degree" : (1,2), "algo__max_iter" : max_iters, "algo__alpha" : alphas}
    model = sklearn.model_selection.GridSearchCV(estimator=model, cv = cross_valid, param_grid=params, n_jobs=5, refit=True, verbose=10)
    
    return model


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()        
    
        model = create_model(train)
        model.fit(train.data, train.target)
        
        for rank, accuracy, params in zip(model.cv_results_["rank_test_score"],
                                         model.cv_results_["mean_test_score"],
                                         model.cv_results_["params"]):
           print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy),
                 *("{}: {:<5}".format(key, value) for key, value in params.items()))
        
        print(model.best_params_)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            print("Saving model...")
            pickle.dump(model, model_file)
        print("Model saved!")

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
#6a81285c-247a-11ec-986f-f39926f24a9c

# Patrik Brocek
# 5ccdc432-238f-11ec-986f-f39926f24a9c

# Martin Oravec
# 1056cfa0-24fb-11ec-986f-f39926f24a9c