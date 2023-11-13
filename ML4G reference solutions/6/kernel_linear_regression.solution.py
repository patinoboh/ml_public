#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.metrics
import sklearn.neural_network

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--epochs", default=200, type=int, help="Number of SGD training epochs")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type (poly/rbf)")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--nn_reference", default=False, action="store_true", help="Show NN solution too")


def main(args: argparse.Namespace) -> tuple[np.ndarray, float, list[float], list[float]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1
    train_data = np.expand_dims(train_data, axis=1)

    test_data = np.linspace(-1.2, 1.2, 2 * args.data_size)
    test_target = np.sin(5 * test_data) + 1
    test_data = np.expand_dims(test_data, axis=1)

    # Initialize the parameters: the betas and the bias.
    betas = np.zeros(args.data_size)
    bias = 0

    def kernel(x, z):
        if args.kernel == "poly":
            return (args.kernel_gamma * x @ z + 1) ** args.kernel_degree
        if args.kernel == "rbf":
            return np.exp(-args.kernel_gamma * (x - z) @ (x - z))

    K = np.array([[kernel(x, z) for x in train_data] for z in train_data])
    Ktest = np.array([[kernel(x, z) for x in train_data] for z in test_data])

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, compute their gradient of the loss, and
        # update the `betas` and the `bias`. You can assume that `args.batch_size`
        # exactly divides `train_data.shape[0]`.
        #
        # We assume the primary formulation of our model prediction is
        #   y = phi(x)^T w + bias,
        # the weights are represented using betas in the dual formulation as
        #   w = \sum_i beta_i phi(x_i),
        # and the loss for a batch $B$ in the primary formulation is the MSE with L2 regularization:
        #   L = 1/|B| * (\sum_{i \in B} 1/2 * (phi(x_i)^T w + bias - t_i)^2) + 1/2 * args.l2 * ||w||^2.
        #
        # You should update the `betas` and the `bias`, so that the update
        # is equivalent to the update in the primary formulation. Be aware that
        # for a single batch, only some betas are updated because of the MSE, but
        # all betas are updated because of L2 regularization.
        #
        # Instead of using the feature map $phi$ directly, we use a given kernel computing
        #   K(x, z) = phi(x)^T phi(z).
        #
        # We consider the following `args.kernel`s:
        # - "poly": K(x, z; degree, gamma) = (gamma * x^T z + 1) ^ degree
        # - "rbf": K(x, z; gamma) = exp^{- gamma * ||x - z||^2}
        # The kernel parameters are specified in `args.kernel_gamma` and `args.kernel_degree`.
        for index in range(0, len(permutation), args.batch_size):
            batch = permutation[index:index + args.batch_size]
            assert len(batch) == args.batch_size

            gradient = (K[batch] @ betas + bias - train_target[batch]) / args.batch_size
            betas -= args.learning_rate * args.l2 * betas  # L2-regularization update
            betas[batch] -= args.learning_rate * gradient  # MSE update
            bias -= args.learning_rate * np.sum(gradient)  # bias update

        # TODO: Append current RMSE on train/test data to `train_rmses`/`test_rmses`.
        train_rmses.append(sklearn.metrics.mean_squared_error(train_target, K @ betas + bias, squared=False))
        test_rmses.append(sklearn.metrics.mean_squared_error(test_target, Ktest @ betas + bias, squared=False))

        if (epoch + 1) % 10 == 0:
            print("After epoch {}: train RMSE {:.2f}, test RMSE {:.2f}".format(
                epoch + 1, train_rmses[-1], test_rmses[-1]))

    if args.plot:
        import matplotlib.pyplot as plt
        # If you want the plotting to work (not required for ReCodEx), compute the `test_predictions`.
        test_predictions = Ktest @ betas + bias

        plt.plot(train_data, train_target, "bo", label="Train target")
        plt.plot(test_data, test_target, "ro", label="Test target")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend()
        if args.nn_reference:
            for params in [{"activation": "tanh"}, {"activation": "relu"}]:
                plt.figure()
                mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(500,), solver="lbfgs", alpha=0, max_iter=2000, **params).fit(np.expand_dims(train_data, -1), train_target)
                plt.plot(train_data, train_target, "bo", label="Train target")
                plt.plot(test_data, test_target, "ro", label="Test target")
                plt.plot(test_data, mlp.predict(np.expand_dims(test_data, -1)), "g-", label="Predictions")
                plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return betas, bias, train_rmses, test_rmses


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    betas, bias, train_rmses, test_rmses = main(args)
    print("Learned betas", *("{:.2f}".format(beta) for beta in betas[:15]), "...")
    print("Learned bias", bias)
