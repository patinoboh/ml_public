#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

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
        #
        # Note that you need to be careful when computing softmax because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate non-positive values, and overflow does not occur.
        def softmax(x):
            # Note that the `axis=-1, keepdims=True` allow processing both 1D and 2D inputs, always
            # computing the softmax "on the last dimension".
            x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return x / np.sum(x, axis=-1, keepdims=True)

        # Manual gradient aggregation variant.
        gradient, gradient_components = 0, 0
        for i in permutation:
            outputs = softmax(train_data[i] @ weights)
            gradient += np.outer(train_data[i], outputs - np.eye(args.classes)[train_target[i]])
            gradient_components += 1
            if gradient_components == args.batch_size:
                weights -= args.learning_rate * gradient / gradient_components
                gradient, gradient_components = 0, 0
        assert gradient_components == 0

        if False:
            # Alternatively, we could process the whole batch at a time, which is more efficient.
            for i in range(0, len(permutation), args.batch_size):
                batch = permutation[i:i + args.batch_size]
                outputs = softmax(train_data[batch] @ weights)
                gradient = train_data[batch].T @ (outputs - np.eye(args.classes)[train_target[batch]]) / len(batch)
                weights -= args.learning_rate * gradient

        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log-likelihood, or cross-entropy loss, or KL loss) per example.
        train_loss = sklearn.metrics.log_loss(train_target, softmax(train_data @ weights))
        train_accuracy = sklearn.metrics.accuracy_score(train_target, np.argmax(train_data @ weights, axis=1))
        test_loss = sklearn.metrics.log_loss(test_target, softmax(test_data @ weights))
        test_accuracy = sklearn.metrics.accuracy_score(test_target, np.argmax(test_data @ weights, axis=1))

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
