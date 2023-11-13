import argparse

import numpy as np
import sklearn.metrics

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
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, float, list[float], list[float]]:

    def K_poly(x, z):
        return ((args.kernel_gamma * np.inner(x, z) + 1) ** args.kernel_degree)

    def K_rbf(x, z):
        return np.exp(-args.kernel_gamma * (np.linalg.norm(x-z)**2))

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

    K = K_rbf
    if args.kernel == "poly":
        K = K_poly

    K_train = np.zeros(shape=[args.data_size, args.data_size])

    K_test = np.zeros(shape=[2 * args.data_size, args.data_size])

    for i in range(args.data_size):
        for j in range(args.data_size):
            K_train[i,j] = K(train_data[i], train_data[j])
            
    for i in range(2*args.data_size):
        for j in range(args.data_size):
            K_test[i,j] = K(test_data[i], train_data[j])

    def pred_train(i):
        res = 0
        for j in range(args.data_size):
            res += betas[j]*K_train[i,j]
        return res+bias



    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        for batch in range(int(train_data.shape[0] / args.batch_size)):

            gradient = np.zeros(shape=betas.shape)
            bias_gradient = 0

            for i in range(args.batch_size):
                i = permutation[ batch * args.batch_size + i ]

                a = pred_train(i) - train_target[i]
                gradient[i] += args.learning_rate * a / args.batch_size
                bias_gradient += args.learning_rate * a / args.batch_size
            
            betas -= gradient + args.learning_rate * args.l2 * betas
            bias -= bias_gradient

        y_test = K_test @ betas + bias
        y_train = K_train @ betas + bias

        

        # TODO: Append current RMSE on train/test data to `train_rmses`/`test_rmses`.
        train_rmses.append(sklearn.metrics.mean_squared_error(y_train, train_target)**(1/2))
        test_rmses.append(sklearn.metrics.mean_squared_error(y_test, test_target)**(1/2))

        if (epoch + 1) % 10 == 0:
            print("After epoch {}: train RMSE {:.2f}, test RMSE {:.2f}".format(
                epoch + 1, train_rmses[-1], test_rmses[-1]))

    if args.plot:
        import matplotlib.pyplot as plt
        # If you want the plotting to work (not required for ReCodEx), compute the `test_predictions`.
        test_predictions = ...

        plt.plot(train_data, train_target, "bo", label="Train target")
        plt.plot(test_data, test_target, "ro", label="Test target")
        # plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return betas, bias, train_rmses, test_rmses


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    betas, bias, train_rmses, test_rmses = main(args)
    print("Learned betas", *("{:.2f}".format(beta) for beta in betas[:15]), "...")
    print("Learned bias", bias)

# 5ccdc432-238f-11ec-986f-f39926f24a9c
# bb29abf7-2547-11ec-986f-f39926f24a9c