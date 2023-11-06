import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn
import sklearn.pipeline
import sklearn.compose
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics

import sklearn.dummy

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
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


def get_model(data):
    int_columns = get_integer_columns(data)
    
    model = [("algo", sklearn.linear_model.LogisticRegression())]

    model = sklearn.pipeline.Pipeline(
        [("preprocess", sklearn.compose.ColumnTransformer(
            [("onehot", sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False), int_columns),
             ("scaler", sklearn.preprocessing.StandardScaler(), ~int_columns),])), 
             ("poly", sklearn.preprocessing.PolynomialFeatures()) ]
        + model
        )
            
    cross_valid = sklearn.model_selection.StratifiedKFold(5)
    params = {'algo__C': [1, 100, 200], 'algo__penalty': ['l2', 'elasticnet'], 'algo__solver': ['liblinear'], 'poly__degree': [2, 3]}
    model = sklearn.model_selection.GridSearchCV(estimator=model, cv = cross_valid, param_grid=params, n_jobs=2, refit=True, verbose=10)
    return model


def test_model(model, test_data, test_target):    
    print(sklearn.metrics.accuracy_score(test_target, model.predict(test_data)))


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        dataset = Dataset()


        # TODO: Train a model on the given dataset and store it in `model`.                 

        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size=0.8)
        #train_data = dataset.data
        #train_target = dataset.target

        model = get_model(train_data)                        
        model.fit(train_data, train_target)
        
        # test_model(model, test_data, test_target)
        
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
