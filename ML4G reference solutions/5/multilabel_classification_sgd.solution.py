#!/usr/bin/env python3
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
parser.add_argument("--with_reference", default=False, action="store_true", help="Use reference implementation")


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    # TODO: The `target` is a list of classes for every input example. Convert
    # it to a dense representation (n-hot encoding) -- for each input example,
    # the target should be vector of `args.classes` binary indicators.
    target = np.zeros([len(target_list), args.classes], dtype=int)
    for row, row_list in zip(target, target_list):
        row[row_list] = 1

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
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        gradient, gradient_components = 0, 0
        for i in permutation:
            gradient += np.outer(train_data[i], sigmoid(train_data[i] @ weights) - train_target[i])
            gradient_components += 1
            if gradient_components == args.batch_size:
                weights -= args.learning_rate * gradient / gradient_components
                gradient, gradient_components = 0, 0
        assert gradient_components == 0

        if False:
            # Alternatively, we could process the whole batch at a time, which is more efficient.
            for i in range(0, len(permutation), args.batch_size):
                batch = permutation[i:i + args.batch_size]
                outputs = sigmoid(train_data[batch] @ weights)
                gradient = train_data[batch].T @ (outputs - train_target[batch]) / len(batch)
                weights -= args.learning_rate * gradient

        # TODO: After the SGD epoch, compute the micro-averaged and the
        # macro-averaged F1-score for both the train test and the test set.
        # Compute these scores manually, without using `sklearn.metrics`.
        assert not np.any(np.isclose(train_data @ weights, 0))
        assert not np.any(np.isclose(test_data @ weights, 0))

        train_pred = train_data @ weights > 0
        test_pred = test_data @ weights > 0

        def f1_score(tp, fp, fn):
            return 2 * tp / (2 * tp + fp + fn) if tp else 0
        def f1_micro(target, pred):
            return f1_score(np.sum((target != 0) & (target == pred)),
                            np.sum((pred != 0) & (target != pred)),
                            np.sum((target != 0) & (target != pred)))
        def f1_macro(targets, preds):
            return np.mean([f1_micro(target, pred) for target, pred in zip(targets.T, preds.T)])

        train_f1_micro = f1_micro(train_target, train_pred)
        train_f1_macro = f1_macro(train_target, train_pred)
        test_f1_micro = f1_micro(test_target, test_pred)
        test_f1_macro = f1_macro(test_target, test_pred)

        if args.with_reference:
            train_f1_micro = sklearn.metrics.f1_score(train_target, train_pred, average="micro", zero_division=0)
            train_f1_macro = sklearn.metrics.f1_score(train_target, train_pred, average="macro", zero_division=0)
            test_f1_micro = sklearn.metrics.f1_score(test_target, test_pred, average="micro", zero_division=0)
            test_f1_macro = sklearn.metrics.f1_score(test_target, test_pred, average="macro", zero_division=0)

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
