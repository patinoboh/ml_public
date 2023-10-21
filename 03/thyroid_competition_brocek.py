import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import sklearn
from sklearn import model_selection, preprocessing, compose, linear_model, pipeline


import numpy as np
import numpy.typing as npt

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
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def create_model(dataset):
    dataset.data = np.insert(dataset.data, dataset.data.shape[1], values=1, axis=1)
    t_size = 0.1
    train_data , test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size=t_size)    
    one_hot_transformer = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore")    
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    polynomial_features = sklearn.preprocessing.PolynomialFeatures()    
    scaler_transformer = sklearn.preprocessing.StandardScaler()        
    
    col_transf = sklearn.compose.ColumnTransformer(transformers = [ ("one hot", one_hot_transformer, list(range(16)) ) ], remainder="passthrough")
    algo = sklearn.linear_model.LogisticRegression(random_state=args.seed)        
    
    potrubie = sklearn.pipeline.Pipeline( [ ("one hot columns", col_transf), ("minmax", min_max_scaler), ("scaler", scaler_transformer) , ("poly", polynomial_features), ("algo", algo) ])
    
    cross_valid = sklearn.model_selection.StratifiedKFold(5)
    params = {"poly__degree" : (1, 2), "algo__solver" : ("lbfgs", "sag"), "algo__C" : [0.01, 1.0, 50.0, 100.0 ]}
    model = sklearn.model_selection.GridSearchCV(estimator=potrubie, cv = cross_valid, param_grid=params, n_jobs=-1, refit=True)    
    
    model.fit(dataset.data, dataset.target)
    return model


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        model = create_model(train)        

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        
        test_data = np.insert(test.data, test.data.shape[1], values=1, axis=1)
        predictions = model.predict(test_data)        

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

# 5ccdc432-238f-11ec-986f-f39926f24a9c
# 0cc3ac3d-24fb-11ec-986f-f39926f24a9c
