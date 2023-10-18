import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.compose
import sklearn.linear_model

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

    The target variable is the number of rentals in the given hour.
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

def get_pipeline():
    one_hot_transformer = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore")    
    scaler_transformer = sklearn.preprocessing.StandardScaler()        
    transf = sklearn.compose.ColumnTransformer(transformers = [ ("one hot", one_hot_transformer, [0,1,2,3,4,5,6] ) ], remainder="passthrough")
    polynomial_features = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
    pipeline = sklearn.pipeline.Pipeline([ ("one hot, scaler transformers", transf), ("polynomial features", polynomial_features) ])
    # pipeline = sklearn.pipeline.Pipeline([ ("one hot, scaler transformers", transf) ])
    return pipeline

def predict_and_print(model, valid_set, valid_target):
    predictions = model.predict(valid_set)
    print(sklearn.metrics.mean_squared_error( predictions, valid_target, squared = False))

def train_model(model, train, pipeline, t_size):    
    train.data = np.insert(train.data, train.data.shape[1], values=1, axis=1)
    train_set, valid_set, train_target, valid_target = sklearn.model_selection.train_test_split(train.data, train.target, test_size=t_size)        
    train_data_processed = pipeline.fit_transform(train_set)
    model.fit(train_data_processed, train_target)
    predict_and_print(model, pipeline.transform(valid_set), valid_target)
    

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    pipeline = get_pipeline()
    
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        
        # TODO: Train a model on the given dataset and store it in `model`.
        
        train.data = np.insert(train.data, train.data.shape[1], values=1, axis=1)        
        model = sklearn.linear_model.Ridge(alpha = 1)                        
        model.fit(pipeline.fit_transform(train.data), train.target)
        
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
        test_data = pipeline.fit_transform(test_data)
        predictions = model.predict(test_data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    
# Patrik Brocek
# znovu odovzdanie z minuleho roka