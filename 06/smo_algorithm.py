import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type (poly/rbf)")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Stopping condition")
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--tolerance", default=1e-7, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.


def kernel(args: argparse.Namespace, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    if(args.kernel == "poly"):
        return (args.kernel_gamma * (x.T @ z + 1)) ** args.degree
    else:
        return np.exp(-args.kernel_gamma * (np.linalg.norm(x-z)**2))    

# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.            


def smo(args: argparse.Namespace,train_data: np.ndarray, train_target: np.ndarray,test_data: np.ndarray, test_target: np.ndarray):
    # Create initial weights.
            
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    # computing kernels for train and test set
    K_train = np.zeros(shape=[args.data_size, args.data_size])
    K_test = np.zeros(shape=[2 * args.data_size, args.data_size])            
    for i in range(args.data_size):
        for j in range(args.data_size):
            K_train[i,j] = kernel(train_data[i], train_data[j])
            
    for i in range(2*args.data_size):
        for j in range(args.data_size):
            K_test[i,j] = kernel(test_data[i], train_data[j])


    passes_without_as_changing = 0
    train_accs, test_accs = [], []
    
    def y_test(dato_index):
        pred = 0
        for i in range(len(test_data[dato_index])):
            pred += a[i] * test_target[i]* K_test[dato_index, i] + b
        return pred
    
    def y(dato_index):
        pred = 0
        for i in range(len(train_data[dato_index])):
            pred += a[i] * train_target[i]* K_train[dato_index, i] + b
        return pred

    def E(dato_index):
        return y(dato_index) - train_target(dato_index)
    
    def check_kkt_conditions(dato_index):
        part1 = a[dato_index] < args.C and train_target[dato_index] * E(dato_index) < -1 * (args.tolerance)
        part2 = a[dato_index] > args.C and train_target[dato_index] * E(dato_index) > args.tolerance
        return (part1 or part2)
    
    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data.
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i.
            j = j + (j >= i)

            # TODO: Check that a[i] fulfills the KKT conditions, using `args.tolerance` during comparisons.
            if(check_kkt_conditions(i)):
                if(2*K_train[i,j] - K_train[i,i] - K_train[j,j] < 0):                    
                    aj_new = a[j] - train_target[j] * ( E(i) - E(j) ) / ( 2*K_train[i,j] - K_train[i,i] - K_train[j,j] ) # compute the updated unclipped a_j^new
                    old_aj = a[j]
                    aj_new = max(0, min(aj_new, args.C)) # clip the a_j^new to suitable [L, H]
                    if(np.abs(aj_new - a[j]) > args.tolerance):
                        a[j] = aj_new
                        old_ai = a[i]
                        ai_new = a[i] - train_target[i] * train_target[j] ( a[j] - old_aj) # a_i^new
                        a[i] = a[i] - train_target[i] * train_target[j] ( a[j] - old_aj) # a_i^new
                        bj_new = b - E(j) - train_target[i]*(a[i] - old_ai) * K_train[i,j] - train_target[j] * (a[j] - old_aj) * K_train[j, j]
                        bi_new = b - E(j) - train_target[i]*(a[i] - old_ai) * K_train[i,i] - train_target[j] * (a[j] - old_aj) * K_train[j, i]
                        if( ai_new > 0 and ai_new < args.C):
                            b = bi_new
                        elif( aj_new > 0 and aj_new < args.C):
                            b = bj_new
                        else:
                            b = (bi_new + bj_new)/2
                        as_changed += 1

            # If the conditions do not hold, then:
            # - compute the updated unclipped a_j^new.
            #
            #   If the second derivative of the loss with respect to a[j]
            #   is > -`args.tolerance`, do not update a[j] and continue
            #   with next i.

            # - clip the a_j^new to suitable [L, H].
            #
            #   If the clipped updated a_j^new differs from the original a[j]
            #   by less than `args.tolerance`, do not update a[j] and continue
            #   with next i.

            # - update a[j] to a_j^new, and compute the updated a[i] and b.
            #
            #   During the update of b, compare the a[i] and a[j] to zero by
            #   `> args.tolerance` and to C using `< args.C - args.tolerance`.

            # - increase `as_changed`.

        # TODO: After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.
                
        successes = 0
        for i in range(len(train_data.shape)):
            successes += (y(i) / train_target[i]) > 0
        train_accs.append(successes / len(train_data))
        successes = 0
        for i in range(len(test_data)):
            successes += (y_test(i) / test_target[i]) > 0
        test_accs.append(successes / len(test_data))

        # Stop training if `args.max_passes_without_as_changing` passes were reached.
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
    support_vectors, support_vector_weights = ...

    print("Done, iteration {}, support vectors {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), len(support_vectors), 100 * train_accs[-1], 100 * test_accs[-1]))

    return support_vectors, support_vector_weights, b, train_accs, test_accs


def main(args: argparse.Namespace):
    # Generate an artificial regression dataset, with +-1 as targets.
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm.
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt

        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap="RdBu")
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap="RdBu", zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#0d0")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap="RdBu", zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ff0")
            plt.legend(loc="upper center", ncol=4)

        # If you want plotting to work (not required for ReCodEx), you need to
        # define `predict_function` computing SVM value `y(x)` for the given x.
        def predict_function(x):
            pred = 0
            for i in range():
                pred += a[i] * train_target[i]* K_train[dato_index, i] + b
            return pred

        plot(predict_function, support_vectors)
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)