#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import sklearn.multioutput
import sklearn.linear_model
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing
import sklearn.neural_network

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization_dictionary.model", type=str, help="Model path")


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8-sig") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants

    def mostLikely(self, nodia: str, preds: list[list[float]]) -> str:
        if preds == []:
            return nodia
        
        indices = []
        for i in range(len(nodia)):
            if nodia[i].lower() in "acdeinorstuyz":
                indices.append(i)
        
        if nodia not in self.variants:
            res = ""
            pred_i = 0
            for i in range(len(nodia)):
                if i in indices:
                    res += applyDia(nodia[i], preds[pred_i])
                    pred_i += 1
                else:
                    res += nodia[i]
            return res
        

        feasible = self.variants[nodia]
        probs = [0.]*len(feasible)
        for i in range(len(preds)):
            for w in range(len(feasible)):
                probs[w] += preds[i][getClass(feasible[w][indices[i]])]

        return feasible[max(range(len(probs)), key=(lambda i: probs[i]))]



def getClass(c: str):
        assert len(c) == 1
        c = c.lower()
        if c in "áéíóúý":
            return 1
        if c in "čďěňřšťůž":
            return 2
        return 0

def applyDia(c: str, pred: list[float]):
    pred = max(range(3), key=(lambda i: pred[i]))

    if pred == 0:
        return c
    elif pred == 1 and c.lower() in "aeiouy":
        trans = str.maketrans("aeiouyAEIOUY", "áéíóúýÁÉÍÓÚÝ")
    elif pred == 2 and c.lower() in "cdenrstuz":
        trans = str.maketrans("cdenrstuzCDENRSTUZ", "čďěňřšťůžČĎĚŇŘŠŤŮŽ")
        
    return c.translate(trans)


class DictModel:

    def __init__(self, window: int, n_grams: int):
        self.n_g = n_grams
        self.w = window + (1 - window % 2)
        
        self.est = make_pipeline(
            sklearn.preprocessing.OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2  
            ),
            sklearn.linear_model.LogisticRegression(
                verbose=100,
                solver="saga",
                multi_class="multinomial",
            )
        )

    def fit(self, data: str, target: str): 
        self.est.fit(self.makeFeatures(data),self.makeTargets(target))
    

    def predict(self, data: str):
        return self.applyPred(data, self.est.predict_log_proba(self.makeFeatures(data)))
    
    def applyPred(self, data: str, pred):
        dct = Dictionary()

        pred_i = 0
        res = ""
        buffer = ""
        preds_buffer = []
        for c in data:
                if c == " ":
                    res += dct.mostLikely(buffer, preds_buffer)+" "
                    buffer = ""
                    preds_buffer = []                        
                else:
                    if c.lower() in "acdeinorstuyz":
                        preds_buffer.append(pred[pred_i])
                        pred_i += 1
                        buffer += c
                    else:
                        buffer += c

        res += dct.mostLikely(buffer, preds_buffer)
        return res
   

    def makeTargets(self, target: str):
        allTargets = []

        for i in range(len(target)):
            if target[i].lower() in "acdeinorstuyzáčďéěíňóřšťúůýž":        
                allTargets.append(getClass(target[i].lower()))

        return np.array(allTargets)

    def makeFeatures(self, data: str):
        allFeatures = []
        for spl in range(len(data)):
        #po pismenkach
            if data[spl].lower() in "acdeinorstuyz":
                
                features = []

                for f in range(self.w):
                #okolo pismenka
                    pos = spl - (self.w-1)//2 + f
                
                    if (pos < 0 or pos >= len(data)):
                        features.append(" ")
                    elif data[pos].lower() not in "abcdefghijklmnopqrstuvwxyz":
                        features.append(" ")
                    else:
                        features.append(data[pos].lower())
                
                for s in range(2, min(self.w, self.n_g)+1):
                    for o in range(-s+1, 1):
                        features.append(data[spl+o:spl+o+s].lower())
                    
                allFeatures.append(features)
        return np.array(allFeatures)



def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Train a model on the given dataset and store it in `model`.
        model = DictModel(17,7)
        model.fit(train.data,train.target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model: DictModel = pickle.load(model_file)

        # Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)