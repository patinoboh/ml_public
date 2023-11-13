import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
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

def kolko_ich_je(data):
    vsetky_pismenka_ktore_mozu_mat_zmenu = "acdeeinorstuuyzáčďéěíňóřšťúůýž"
    pocet_zmenitelnych_pismen = 0
    for pismenko in data:
        if(pismenko in vsetky_pismenka_ktore_mozu_mat_zmenu):
            pocet_zmenitelnych_pismen += 1
    return pocet_zmenitelnych_pismen

def blabla(data):
    # je tam 0 ak nema nic
    # 1 ak ma dlzen
    # 2 makcen
    # 3 kruzok
    
    cisto_mekcenove_pismenka = "řžďčšňť"
    mekcenove_pismenka_aj_bez_mekcenov = "rřzždďcčsšnňtť"
    cisto_dlznove_pismenka = "áíýó"
    dlznove_pismenka_aj_bez_dlznov = "aáiíyýoó"
    specialne_obojake_pismenka = "eéěuúů"
    
    targets = np.zeros((kolko_ich_je(data), 1)) # pre je jeho target ci ma na tom mieste makcen, dlzen, kruzok alebo nic    
    vsetky_pismenka_ktore_mozu_mat_zmenu = "acdeeinorstuuyzáčďéěíňóřšťúůýž"
    counter = 0
        
    for pismenko in data:    
        if(pismenko in vsetky_pismenka_ktore_mozu_mat_zmenu):
            if(pismenko in mekcenove_pismenka_aj_bez_mekcenov):
                targets[counter][0] = 2 if pismenko in cisto_mekcenove_pismenka else 0
            elif(pismenko in dlznove_pismenka_aj_bez_dlznov):
                targets[counter][0] = 1 if pismenko in cisto_dlznove_pismenka else 0
            elif(pismenko in specialne_obojake_pismenka):
                targets[counter][0] = 1 if pismenko in "éú" else 0
                targets[counter][0] = 2 if pismenko in "ě" else targets[counter][0]
                targets[counter][0] = 3 if pismenko in "ů" else targets[counter][0]
                targets[counter][0] = 0 if pismenko in "eu" else targets[counter][0]                
            counter += 1
    return targets

def blabla_odzadu(data, predictions):
    
    return


    #         if(pismenko in "rř"):
    #             targets[counter][1] = 1 if pismenko == "ř" else 0
    #             targets[counter][3] = 0 if pismenko == "ř" else 1
    #         elif(pismenko in "zž"):
    #             targets[counter][1] = 1 if pismenko == "ř" else 0
    #             targets[counter][3] = 0 if pismenko == "r" else 1
    #         elif(pismenko in "dď"):
    #             targets[counter][1] = 1 if pismenko == "ď" else 0
    #             targets[counter][3] = 0 if pismenko == "ď" else 1
    #         elif(pismenko in "cč"):
    #             targets[counter][1] = 1 if pismenko == "č" else 0
    #             targets[counter][3] = 0 if pismenko == "č" else 1
    #         elif(pismenko in "sš"):
    #             targets[counter][1] = 1 if pismenko == "š" else 0
    #             targets[counter][3] = 0 if pismenko == "š" else 1
    #         elif(pismenko in "nň"):
    #             targets[counter][1] = 1 if pismenko == "ň" else 0
    #             targets[counter][3] = 0 if pismenko == "ň" else 1
    #         elif(pismenko in "tť"):
    #             targets[counter][1] = 1 if pismenko == "ť" else 0
    #             targets[counter][3] = 0 if pismenko == "ť" else 1
    #         elif(pismenko in "aá"):
    #             targets[counter][0] = 1 if pismenko == "á" else 0
    #             targets[counter][3] = 0 if pismenko == "á" else 1
    #         elif(pismenko in "ií"):
    #             targets[counter][0] = 1 if pismenko == "í" else 0
    #             targets[counter][3] = 0 if pismenko == "í" else 1
    #         elif(pismenko in "yý"):
    #             targets[counter][0] = 1 if pismenko == "ý" else 0
    #             targets[counter][3] = 0 if pismenko == "ý" else 1
    #         elif(pismenko in "oó"):
    #             targets[counter][0] = 1 if pismenko == "ó" else 0
    #             targets[counter][3] = 0 if pismenko == "ó" else 1
    #         elif(pismenko in "eéě"):
    #             targets[counter][0] = 1 if pismenko == "é" else 0
    #             targets[counter][1] = 1 if pismenko == "ě" else 0
    #             targets[counter][3] = 0 if pismenko == "éě" else 1
    #         elif(pismenko in "uúů"):
    #             targets[counter][0] = 1 if pismenko == "ú" else 0
    #             targets[counter][2] = 1 if pismenko == "ů" else 0
    #             targets[counter][3] = 0 if pismenko == "úů" else 1
    #         counter += 1
    # return targets


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        model = ...

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = ...

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)