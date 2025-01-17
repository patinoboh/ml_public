#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", choices=["gaussian", "multinomial", "bernoulli"])
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute the probability density function
    #   of a Gaussian distribution using `scipy.stats.norm`, which offers
    #   `pdf` and `logpdf` methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.
    #
    # In all cases, the class prior is the distribution of the train data classes.

    prior = np.bincount(train_target) / len(train_target)
    log_prob = np.zeros((len(test_data), args.classes)) + np.log(prior)

    if args.naive_bayes_type == "gaussian":
        param = {
            "mean": np.zeros((train_data.shape[1], args.classes)),
            "std": np.zeros((train_data.shape[1], args.classes)),
        }
        for i in range(args.classes):
            class_indicies = train_target == i
            param["mean"][:,i] = np.mean(train_data[class_indicies], axis=0)
            param["std"][:,i] = np.sqrt(np.var(train_data[class_indicies], axis=0) + args.alpha)

        log_prob += np.sum(scipy.stats.norm(loc=param["mean"], scale=param["std"]).logpdf(test_data[:, :, np.newaxis]), axis=1)

    elif args.naive_bayes_type == "multinomial":
        param = {
            "p": np.zeros((train_data.shape[1], args.classes)),
        }
        for i in range(args.classes):
            class_indicies = train_target == i
            param["p"][:,i] = np.sum(train_data[class_indicies], axis=0) + args.alpha
            param["p"][:,i] /= np.sum(param["p"][:,i])
    
        log_prob += np.matmul(test_data, np.log(param["p"]))

    elif args.naive_bayes_type == "bernoulli":
        train_data = train_data >= 8
        param = {
            "p": np.zeros((train_data.shape[1], args.classes)),
        }
        for i in range(args.classes):
            class_indicies = train_target == i
            param["p"][:,i] = (np.sum(train_data[class_indicies], axis=0) + args.alpha) / (len(train_data[class_indicies]) + 2 * args.alpha)

        test_data = test_data >= 8
        log_prob += np.matmul((test_data != 0), np.log(param["p"])) + np.matmul((test_data == 0), np.log(1 - param["p"]))    

    # TODO: Predict the test data classes, and compute
    # - the test set accuracy, and
    # - the joint log-probability of the test set, i.e.,
    #     \sum_{(x_i, t_i) \in test set} \log P(x_i, t_i).
    test_accuracy = sklearn.metrics.accuracy_score(test_target, np.argmax(log_prob, axis=1))
    test_log_probability = np.sum(log_prob[np.arange(len(test_target)), test_target])

    return 100 * test_accuracy, test_log_probability


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy, test_log_probability = main(args)

    print("Test accuracy {:.2f}%, log probability {:.2f}".format(test_accuracy, test_log_probability))

#Rasto Nowak

#6a81285c-247a-11ec-986f-f39926f24a9c

#Patrik Brocek

#5ccdc432-238f-11ec-986f-f39926f24a9c

#Martin Oravec

#1056cfa0-24fb-11ec-986f-f39926f24a9c
