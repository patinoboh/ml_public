#MANUALNE ZAKURVI SLOVA NA RETAZCE INTOV A NA TO HODI NEURONKU

import os
import sys
import urllib

import numpy as np

import sklearn.preprocessing
import sklearn.pipeline
import sklearn.neural_network
import sklearn.model_selection
#jebem
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


def maxInd(arr):
    ind = 0
    for i in range(len(arr)):
        ind = i if arr[i] > arr[ind] else ind
    return ind

class DictModel:

    def __init__(self, window: int):
        self.w = window + (1 - window % 2) #chceme neparne velke okno

        self.est = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(300,200,100))


    def fit(self, data: str, target: str):
        self.est.fit(self.makeFeatures(data), self.makeTargets(target))
        #ezz
        words = data.split(' ')
        golds = target.split(' ')
        self.dictionary = {words[i]: golds[i] for i in range(len(words))}
    

    def predict(self, data: str):
        outputLines = []
        for line in data.split('\n'):
            line_feat = self.makeFeatures(line)
            line_pred = [maxInd(dist) for dist in self.est.predict(line_feat)]
            outputLines.append(self.processLine(line, line_pred))
        return '\n'.join(outputLines)
    
    #skurvnea picovina
    def processLine(self, line, line_pred):
        outWords = []
        pred_i = 0
        dict_buffer = ""
        nnet_buffer = ""
        for c in line+" ":
            if c != ' ':
              #kokotina yjebana
                if c in "acdeinorstuyz":
                    nnet_buffer += self.applyDia(c, line_pred[pred_i])
                    pred_i += 1
                else:
                    nnet_buffer += c
                dict_buffer += c
            else:
              #zpicena chujovina
                if dict_buffer in self.dictionary:
                    outWords.append(self.dictionary[dict_buffer])
                else:
                    outWords.append(nnet_buffer)
                dict_buffer = ""
                nnet_buffer = ""
                
        return " ".join(outWords)

    #chujpvina
    def applyDia(self, c: str, pred: int):
        if pred == 0:
            return c
        
        elif pred == 1 and c.lower() in "aeiouy":
            trans = str.maketrans("aeiouyAEIOUY", "áéíóúýÁÉÍÓÚÝ")
        
        elif pred == 2 and c.lower() in "cdenrstz":
            trans = str.maketrans("cdenrstzCDENRSTZ", "čďěňřšťžČĎĚŇŘŠŤŽ")

        elif pred == 3 and c.lower() == 'u':
            trans = str.maketrans("uU", "ůŮ")
        
        else:
            trans = str.maketrans("","")
        #transformacie tykkt
        return c.translate(trans)

#hovno
    def predictOne(self, word):
        if word in self.dictionary:
            return self.dictionary[word]
        
    #dolzeita hcujpica
    def getClass(self, c: str):
        assert len(c) == 1
        c = str.lower(c)
        if c in "áéíóúý":
            return np.array([0,1,0,0])
        if c in "čďěňřšťž":
            return np.array([0,0,1,0])
        if c == "ů":
            return np.array([0,0,0,1])
        return np.array([1,0,0,0])
    
#no to ma pojeb
    def makeTargets(self, target: str):
        allTargets = []

        for i in range(len(target)):
            if target[i].lower() in "acdeinorstuyzáčďéěíňóřšťúůýž":        
                allTargets.append(self.getClass(target[i].lower()))

        return np.array(allTargets)

    @staticmethod
    def oneHotPismenko(i):
        res = np.zeros(27)
        res[i] = 1
        return res

    #ako vazne
    def makeFeatures(self, data: str):
        allFeatures = []
        for line in data.split('\n'):
            for spl in range(len(line)):
          #po pismenkach
                if line[spl].lower() in "acdeinorstuyz":
                    
                    features = []

                    for f in range(self.w):
                    #okolo pismenka jak kokoto
                        pos = spl - (self.w-1)//2 + f
                    
                        if (pos < 0 or pos >= len(line)):#uuuuplny kokot
                            features.append(self.oneHotPismenko(26))
                        if data[pos].lower() not in "abcdefghijklmnopqrstuvwxyz":
                            features.append(self.oneHotPismenko(26))
                        else:#to vis
                            features.append(self.oneHotPismenko(ord(line[spl].lower())-ord("a")))
                    
                    allFeatures.append(features)
                    
        return np.array(allFeatures)
              #a zajeb ho tam!!



model = DictModel(11)

dataset = Dataset()

train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
    dataset.data.split('\n'),
    dataset.target.split('\n'),
    test_size = 0.1
)


model.fit("\n".join(train_data),"\n".join(train_target))

pred = model.predict('\n'.join(test_data))

words = pred.split(' ')
golds = ("\n".join(test_target)).split(" ")

#vypise kokotiny
for i in len(test_target.split(' ')):
    print(words[i],golds[i])


