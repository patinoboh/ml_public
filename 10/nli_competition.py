#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
from typing import Optional

import numpy as np
import numpy.typing as npt
import sklearn
import sklearn.pipeline
import sklearn.svm
import sklearn.feature_extraction

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

        # TODO: Train a model on the given dataset and store it in `model`.
        
        algo = sklearn.svm.LinearSVC()
        
        words = sklearn.feature_extraction.text.TfidfVectorizer()
        chars = sklearn.feature_extraction.text.TfidfVectorizer()
        
        #model = sklearn.pipeline.Pipeline([
         #   ("features_words", words),
          #  ("features_chars", chars),
           # ("algo", algo)])
           
        model = sklearn.pipeline.Pipeline([
            ("features", sklearn.pipeline.FeatureUnion(
                ([("words", sklearn.feature_extraction.text.TfidfVectorizer())]) + 
                ([("chars", sklearn.feature_extraction.text.TfidfVectorizer())]))),
            ("algo", algo)])
        
        # TODO CV
        
        # , "features__words__stop_words" : [None], "features__chars__stop_words" : [None]
        grid = {"features__words__sublinear_tf" : [True], "features__chars__sublinear_tf" : [False],"features__words__lowercase" : [True],
                "features__chars__lowercase" : [False], "features__words__analyzer" : ["word"], "features__chars__analyzer" : ["char_wb"],
                #"features__words__ngram_range" : [(1, 2), (1, 3), (1, 4)], "features__chars__ngram_range" : [(1, 2), (1, 3), (1, 4)],
                # toto bolo 0.819
                "features__words__ngram_range" : [(1, 2)], "features__chars__ngram_range" : [ (1, 3), (1, 4), (1,5), (1,6)],
                # toto bolo TODO
                "features__words__binary" : [False], "features__chars__binary" : [True]
                }
        model = sklearn.model_selection.GridSearchCV(estimator=model,  
                                                      cv = sklearn.model_selection.StratifiedKFold(5), 
                                                      param_grid=grid, 
                                                      n_jobs=7, 
                                                      refit=True,
                                                      verbose =100)
        

        
        model.fit(train.data, train.target)
        
        print("Best parameters",model.best_params_)
        print("Best Score:", model.best_score_)

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
    
    
#Rasto Nowak

#6a81285c-247a-11ec-986f-f39926f24a9c

#Patrik Brocek

#5ccdc432-238f-11ec-986f-f39926f24a9c

#Martin Oravec

#1056cfa0-24fb-11ec-986f-f39926f24a9c
