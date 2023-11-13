import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import sklearn
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.pipeline

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=True, type=str, help="Path to the dataset to predict")
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


def create_model():
    standard_scaler = sklearn.preprocessing.StandardScaler()
    algo = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(40,20))

    cross_valid = sklearn.model_selection.StratifiedKFold(4)

    pipe = sklearn.pipeline.Pipeline(
        [("scaler",standard_scaler),("algo",algo)]
    )

    params ={
        "algo__solver":("lbfgs","sag"),
        "algo__alpha":[0.00001,0.0004],
        "algo__learning_rate":["adaptive","invscaling"]
    }

    return sklearn.model_selection.GridSearchCV(estimator=pipe, cv=cross_valid, param_grid=params, n_jobs=-1, refit=True, verbose=100)


def char_to_int(a):
    if not a.isalpha():
        return 0
    a = a.lower()
    return ord(a) - ord("a") + 1

def kolko_ich_je(data):
    vsetky_pismenka_ktore_mozu_mat_zmenu = "acdeeinorstuuyzáčďéěíňóřšťúůýž"
    pocet_zmenitelnych_pismen = 0
    for pismenko in data:
        if(pismenko.lower() in vsetky_pismenka_ktore_mozu_mat_zmenu):
            pocet_zmenitelnych_pismen += 1
    return pocet_zmenitelnych_pismen

def get_data(data):
        
    new_size = kolko_ich_je(data)
    zajimave = "acdeinorstuyz"
    # indexes = np.zeros(shape=[new_size])
    indexes = np.zeros(shape=[new_size]).astype(int)

    win_size = 11
    pad = int((win_size - 1) / 2)
    res = np.zeros(shape=[new_size,win_size])
    window = [0] * win_size

    for i in range(pad + 1):
        window[pad + i] = char_to_int(data[i])
    
    row_count = 0
    
    for i in range(int(len(data))):
        if data[i].lower() in zajimave:
            res[row_count] = np.array(window).reshape(1,-1)
            indexes[row_count] = i
            row_count += 1
            #print(res[i])
        
        for j in range(win_size - 1):
            window[j] = window[j + 1]
        if pad + i + 1 < len(data):
            window[-1] = char_to_int(data[pad + i + 1])
        else:
            window[-1] = 0
    
    return res, indexes

def get_targets(data):
    # je tam 0 ak nema nic
    # 1 ak ma dlzen
    # 2 makcen
    # 3 kruzok
    
    makcene = "řžďčšňěť"
    dlzne = "áíéýúó"
    
    targets = np.zeros((kolko_ich_je(data))) # pre je jeho target ci ma na tom mieste makcen, dlzen, kruzok alebo nic    
    vsetky_pismenka_ktore_mozu_mat_zmenu = "acdeeinorstuuyzáčďéěíňóřšťúůýž"
    counter = 0
        
    for pismenko in data:    
        if(pismenko in vsetky_pismenka_ktore_mozu_mat_zmenu):
            if(pismenko in makcene):
                targets[counter] = 2 
            elif(pismenko in dlzne):
                targets[counter] = 1 
            elif(pismenko == "ů"):
                targets[counter] = 3
            else:
                targets[counter] = 0               
            counter += 1
    return targets



def most_probable_form_of_char(char_to_modify : str, probabilities) -> int:
    cisto_mekcenove_pismenka = "řžďčšňť"
    cisto_mekcenove_pismenka_bez_hacku = "rzdcsnt"
    cisto_dlznove_pismenka = "áíýó"
    cisto_dlznove_pismenka_bez_carek = "aiyo"

    char = char_to_modify.lower()
    return_char = char
    # probabilities[0] -> p0
    # probabilities[1] -> p1
    # probabilities[2] -> p2
    # probabilities[3] -> p3
    p0 = 1 if probabilities == 0 else 0
    p1 = 1 if probabilities == 1 else 0
    p2 = 1 if probabilities == 2 else 0
    p3 = 1 if probabilities == 3 else 0
    print(p0, p1, p2, p3)
    
    if char in cisto_mekcenove_pismenka_bez_hacku and p2 > p0:
        return_char = cisto_mekcenove_pismenka[cisto_mekcenove_pismenka_bez_hacku.index(char)]
    elif char in cisto_dlznove_pismenka_bez_carek and p1 > p0:
        return_char = cisto_dlznove_pismenka[cisto_dlznove_pismenka_bez_carek.index(char)]
    elif char == 'u':
        if p1 > p3 and p1 > p0:
            return_char = 'ú'
        if p3 > p1 and p3 > p0:
            return_char = 'ů'
    elif char == "e":
        if p1 > p2 and p1 > p0:
            return_char = 'é'
        if p2 > p1 and p2 > p0:
            return_char = 'ě'
    else:
        return "."
        print("Chyba v metrixu.", end=" ")        

    if char_to_modify.isupper():
        return_char = return_char.upper()
    return return_char

def diacritize(input_text : str,
               operation_indeces : np.ndarray,
               operations: np.ndarray,
               predications : np.ndarray) -> str:
    if operations.shape[0] != operation_indeces.shape[0]:
        return ''

    list_of_chars = list(input_text)
    for i in range(operations.shape[0]):
        swap_index = int(operation_indeces[i])
        char_to_modify = list_of_chars[swap_index]
        list_of_chars[swap_index] = most_probable_form_of_char(char_to_modify, predications[i])

    return ''.join(list_of_chars)


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        train_data, _ = get_data(train.data)
        train_targets = get_targets(train.target)
      

        model = create_model().fit(train_data, train_targets)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        # test = Dataset(args.predict)
        test = Dataset()

        data, operation_indices = get_data(test.data)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        
        predictions = model.predict(data)
        
        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = diacritize(test.data, operation_indices, predictions , predictions )
        print(predictions)
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)