
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def softmax(x, axis = 0):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)  # todo: check axis

def softmax_mat(X):
    return np.array([softmax(i) for i in X])

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def most_probable(inputs):
        ind = 0
        #print(y)
        for i in range(len(inputs)):
            ind = ind if inputs[ind] > inputs[i] else i
        return ind

def macro_f1(predictions):
    macro = 0
    for row in predictions.T:
        tp, tn, fn, fp = 0, 0, 0, 0
        for element in row:
            if(element == 0):
                tn += 1
            elif(element == 1):
                fn += 1
            elif(element == 2):
                fp += 1
            elif(element == 3):
                tp +=1
        macro += (2*tp) / (2*tp + fp + fn)
    macro /= predictions.shape[1]
    return macro

def micro_f1(predictions):
    # 0 = TN, 1 = FN, 2 = FP, 3 = TP
    tp, tn, fn, fp = 0, 0, 0, 0
    for row in predictions:
        for element in row:
            if(element == 0):
                tn += 1
            elif(element == 1):
                fn += 1
            elif(element == 2):
                fp += 1
            elif(element == 3):
                tp += 1
    return (2*tp) / (2*tp + fp + fn)

def main(args: argparse.Namespace):
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    # TODO: The `target` is a list of classes for every input example. Convert
    # it to a dense representation (n-hot encoding) -- for each input example,
    # the target should be vector of `args.classes` binary indicators.    
        
    target = np.zeros((data.shape[0], args.classes))
    for i, classifications in enumerate(target_list):
        for data_classifications in classifications:
            target[i][data_classifications] = 1
        
    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        
        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        ...
        for i in range(0, train_data.shape[0], args.batch_size):
            gradient = 0
            for j in permutation[i:i+args.batch_size]:                
                categorical_probab = sigmoid(train_data[j] @ weights) - train_target[j]
                x = train_data[j]
                gradient += np.outer(categorical_probab, x)
            gradient /= args.batch_size
            weights -= args.learning_rate * gradient.T
    
        predictions = np.round_(sigmoid(train_data @ weights))
        predictions = predictions * 2 + train_target # 0 = TN, 1 = FN, 2 = FP, 3 = TP
        
        test_pred = np.round(sigmoid(test_data @ weights))
        test_pred = test_pred * 2 + test_target
        # TODO: After the SGD epoch, compute the micro-averaged and the
        # macro-averaged F1-score for both the train test and the test set.
        # Compute these scores manually, without using `sklearn.metrics`.
        train_f1_micro, train_f1_macro, test_f1_micro, test_f1_macro = (micro_f1(predictions), macro_f1(predictions), micro_f1(test_pred) , macro_f1(test_pred))

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")

# Patrik Brocek znovu odovzdanie z minuleho roku
# v skupine s Vlado Vozarom a Michal Sevcikom

# 5ccdc432-238f-11ec-986f-f39926f24a9c
# 0cc3ac3d-24fb-11ec-986f-f39926f24a9c
# b2a1ea6c-8a6a-4446-a522-7200d5522a14
