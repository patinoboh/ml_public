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

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
# add binary argument
# parser.add_argument("--pred", default=False, action="store_true", help="Use binary features")
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")


# patrik first try

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
    
    def get_features(self):
        window_size = 5            # to each side
        data = self.data.lower()
        targets = self.target.lower()
        features, target_features = [], []
        for i, letter in enumerate(data):            
            if(letter not in self.LETTERS_NODIA):
                continue # sorry ale fakt sa to inak nedá napísať            
            feature = [letter]
            for offset in list(range(-window_size, 0)) + list(range(1, window_size+1)):
                if(i+offset < 0 or i + offset >= len(data)):
                    feature.append(" ") # TODO co tam appendnut, ak sme na kraji?
                else:
                    feature.append(data[i+offset]) # tu malo byt asi plus lebo ten offset ide od zapornych uz lulw
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
# 3. najebat tam brutal model a eskere
# 4. dorobit dalsie features - pridat n-gramy


def get_predictions(model, data):
    # window_size musi byt hyperparameter aj pre tuto funkciu
    # toto sa mi ale vobec nechce
    # lebo pre prediction toho modelu mu musime poslat ten oneHot list susedov a to je pain
    # ?? da sa dať na model nejaký .fit_transform?
    #hmm nepoznam take  
      
    window_size = 5
    predictions = []
    lower = data.lower()    
    
    lower.append([" "]* window_size)
    lower.insert(0, [" "]* window_size)    
    
    data.append([" "]* window_size)
    data.insert(0, [" "]* window_size)
    

    pred = model.predict(data)
    
    y = 0
    for i in (range(window_size, len(data) - window_size)):
        letter = data[i]
        if(i.lower() not in data.LETTERS_NO_DIA):
            predictions.append(letter)
        else:            
            # prediction_vector =[lower[i]] + list(lower[i - window_size : i + window_size + 1])
            prediction_vector = pred[y]
            y += 1
 #         ORI si mysli ze  by tam mal byt len .transform() a mali by sme mat oddelleny transformer a model teda algoritmus ako  
            prediction_vector = model.predict(prediction_vector) # SNAAAD TODO
            # co teda este s tymto oneHot
            # ten bude ok len musim zmenit aby som cyclil cez tie data s offsetom, dumam a nejde mi to pockaj 
            correction = model.predict(prediction_vector)
            corrected_letter = ""
            if correction == 0:
                predictions.append(letter)
            elif correction == 1:
                basic = "aeiouyAEIOUY"
                dĺžne = "áéíóúýÁÉÍÓÚÝ"
                corrected_letter = dĺžne[basic.index(letter)]
            elif correction == 2:
                normálne_písmen = "cdenrstuzCDENRSTUZ"
                mäkčene_a_vôkáň = "čďěňřšťůžČĎĚŇŘŠŤuŽ" # neviem napísať veľký vokáň (ak existuje)
                corrected_letter = mäkčene_a_vôkáň[normálne_písmen.index(letter)]
            predictions.append(corrected_letter)                    
    
    return predictions


def get_model():


    # model =sklearn.linear_model.LogisticRegression(verbose = 100, solver = 'saga', max_iter = 100)
    model = sklearn.neural_network.MLPClassifier(verbose=100,hidden_layer_sizes=(10), max_iter = 10)

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
        # split the data into train and target
        # features_train, features_test, targets_train, targets_test = sklearn.model_selection.train_test_split(
        #     features, 
        #     targets, test_size=1)
        
        model.fit(features, targets)
        # model.fit(features_train, targets_train)        
        # predictions = model.predict(features_test)
        
        # differences = sum(a != b for a, b in zip(predictions, targets_test))
        # print(differences)    

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        
        test.target = test.data
        test_features, _ = test.get_features()
        predictions = get_predictions(model, test_features)
        
        return predictions

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        
        


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)