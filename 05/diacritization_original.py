#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import sklearn
import sklearn.pipeline
import sklearn.neural_network
import sklearn.linear_model
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
# add binary argument
# parser.add_argument("--pred", default=False, action="store_true", help="Use binary features")
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)
    
    def one_hot(self, letter):
        if letter in "abcdefghijklmnopqrstuvwxyz":
            return np.eye(26)["abcdefghijklmnopqrstuvwxyz".index(letter)]
        else:
            return np.zeros(26)



    def get_features(self):
        window_size = 1            # to each side
        data = self.data.lower()
        targets = self.target.lower()
        features, target_features = [], []
        for i, letter in enumerate(data):            
            if(letter not in self.LETTERS_NODIA):
                continue
            feature = [self.one_hot(letter)]
            for offset in list(range(-window_size, 0)) + list(range(1, window_size+1)):
                if(i+offset < 0 or i + offset >= len(data)):
                    feature.append(self.one_hot(" ")) # TODO co tam appendnut, ak sme na kraji?
                else:
                    feature.append(self.one_hot(letter))
            features.append(feature)
            if targets[i] in "áéíóúý":
                target_features.append(1)
            elif targets[i] in "čďěňřšťůž":
                target_features.append(2)
            else:
                target_features.append(0)
            
        return features, target_features


# TODO 
# 1. neviem ci funguju predictions ale sak to lahko zistime
# 2. upravit window_size aby sme ju mohli menit len na jednom mieste
# 3. dat tam brutal model a eskere
# 4. dorobit dalsie features - pridat n-gramy

def prediction(model,data):
    
    #data.target = data.data    
    
    data.target = data.data
    features, _ = data.get_features()
    
    new_targets = model.predict(features)
    
    predictions = list(data.data)

    basic = "aeiouyAEIOUY"
    dlzne = "áéíóúýÁÉÍÓÚÝ"
    normalne_pismen = "cdenrstuzCDENRSTUZ"
    makcene_a_vokan = "čďěňřšťůžČĎĚŇŘŠŤŮŽ"
    
    index_to_letter_correction = 0
    for i, letter in enumerate(data.data):
        if(letter.lower() not in data.LETTERS_NODIA):
            continue
        corrected_letter = predictions[i]
        if new_targets[index_to_letter_correction] == 1 and letter in basic:
            #print(letter, new_targets[index_to_letter_correction])
            corrected_letter = dlzne[basic.index(letter)]
        elif new_targets[index_to_letter_correction] == 2 and letter in normalne_pismen:
            corrected_letter = makcene_a_vokan[normalne_pismen.index(letter)]
        predictions[i]=corrected_letter 
        index_to_letter_correction += 1
        
    predictions = "".join(predictions)
    return predictions


def get_model():
    # model =sklearn.linear_model.LogisticRegression(verbose = 100, solver = 'saga', max_iter = 100)
    
    ###este tol=0
    
    model = sklearn.neural_network.MLPClassifier(verbose=100,hidden_layer_sizes=(100), max_iter = 50)

    model = sklearn.pipeline.Pipeline([
            ("scaler", sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')),
            ("algo", model),
        ]) 
    return model

def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        model = get_model()                        
        features, targets = train.get_features()        
        model.fit(features, targets)
        
        #pred = prediction(model, train)
        
        # print pred into a file
        #with open("predikcie.txt", "w") as f:
           # for letter in pred:
            #    f.write(letter)

        #differences = sum(a != b for a, b in zip(pred, train.target))
        #print(differences)    
        
        #TOTO ak budeme robit mlp
        model = model.named_steps["algo"]
        model._optimizer = None
        for i in range(len(model.coefs_)): model.coefs_[i] = model.coefs_[i].astype(np.float16)
        for i in range(len(model.intercepts_)): model.intercepts_[i] = model.intercepts_[i].astype(np.float16)
        

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        
        predictions = prediction(model, test)
        
        return predictions

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    