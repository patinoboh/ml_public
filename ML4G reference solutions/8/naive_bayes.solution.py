#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.naive_bayes

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type gaussian/multinomial/bernoulli")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--with_reference", default=False, action="store_true", help="Show also reference results")


def main(args: argparse.Namespace) -> float:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    if args.naive_bayes_type == "bernoulli":
        data = data >= 8

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
    # Fit priors.
    priors = np.bincount(train_target) / len(train_target)

    # Fit the parameters.
    if args.naive_bayes_type == "gaussian":
        params = {"means": np.zeros([train_data.shape[1], args.classes]),
                  "stds": np.zeros([train_data.shape[1], args.classes])}
    elif args.naive_bayes_type in ["bernoulli", "multinomial"]:
        params = {"ps": np.zeros([train_data.shape[1], args.classes])}

    for c in range(args.classes):
        c_data = train_data[train_target == c]
        if args.naive_bayes_type == "gaussian":
            params["means"][:, c] = np.mean(c_data, axis=0)
            params["stds"][:, c] = np.sqrt(np.var(c_data, axis=0) + args.alpha)
        if args.naive_bayes_type == "multinomial":
            params["ps"][:, c] = np.sum(c_data, axis=0) + args.alpha
            params["ps"][:, c] = params["ps"][:, c] / np.sum(params["ps"][:, c])
        if args.naive_bayes_type == "bernoulli":
            params["ps"][:, c] = (np.sum(c_data, axis=0) + args.alpha) / (len(c_data) + 2 * args.alpha)

    # Model prediction.
    log_probabilities = np.zeros((len(test_data), args.classes))
    log_probabilities += np.log(priors)
    if args.naive_bayes_type == "gaussian":
        log_probabilities += np.sum(scipy.stats.norm(loc=params["means"], scale=params["stds"]).logpdf(test_data[:, :, np.newaxis]), axis=1)
    if args.naive_bayes_type == "multinomial":
        log_probabilities += test_data @ np.log(params["ps"])
    if args.naive_bayes_type == "bernoulli":
        log_probabilities += (test_data != 0) @ np.log(params["ps"])
        log_probabilities += (test_data == 0) @ np.log(1 - params["ps"])

    # TODO: Predict the test data classes and compute the test accuracy.
    test_accuracy = sklearn.metrics.accuracy_score(test_target, np.argmax(log_probabilities, axis=1))

    if args.with_reference:
        if args.naive_bayes_type == "gaussian":
            nb = sklearn.naive_bayes.GaussianNB(var_smoothing=args.alpha/np.var(train_data, axis=0).max())
        if args.naive_bayes_type == "multinomial":
            nb = sklearn.naive_bayes.MultinomialNB(alpha=args.alpha)
        if args.naive_bayes_type == "bernoulli":
            nb = sklearn.naive_bayes.BernoulliNB(alpha=args.alpha)
        nb.fit(train_data, train_target)
        test_accuracy = sklearn.metrics.accuracy_score(test_target, nb.predict(test_data))
        print("Scikit-learn test accuracy {:.2f}%".format(
            100 * sklearn.metrics.accuracy_score(test_target, nb.predict(test_data))))

    if args.plot:
        import matplotlib.pyplot as plt
        def plot(data, title, mappable=None):
            plt.subplot(3, 1, 1 + len(plt.gcf().get_axes()) // 2)
            plt.imshow(data, cmap="plasma", interpolation="none")
            plt.axis("off")
            plt.colorbar(mappable, aspect=10)
            plt.title(title)

        H = np.sqrt(data.shape[1]).astype(int)
        data = {key: np.pad(value.reshape([H, H, -1]).transpose([0, 2, 1]).reshape([H, -1]), [(0, 0), (0, H * (10 - args.classes))])
                for key, value in params.items()}
        plt.figure(figsize=(8*1, 0.9*3))
        if args.naive_bayes_type == "gaussian":
            plot(data["means"], "Estimated means")
            plot(data["stds"], "Estimated standard deviations")
            combined = plt.cm.plasma(data["means"] / np.max(data["means"]))
            combined[:, :, 1] = data["stds"] / np.max(data["stds"])
            plot(combined, "Estimated means (R+B) and stds (G)")
        else:
            plot(data["ps"], "Estimated probabilities")
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight", pad_inches=0)

    return test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))
