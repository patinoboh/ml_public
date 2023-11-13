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
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[tuple[np.ndarray, ...], list[float]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    def ReLU(x):
        return np.maximum(0,x)
        #return max(0,x)
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis = -1,keepdims=True))
        softmax = exp_x / np.sum(exp_x, axis= -1, keepdims=True)
        return softmax
    
    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where $ReLU(x) = max(x, 0)$, and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as `ReLU(inputs @ weights[0] + biases[0])`.
        # The value of the output layer is computed as `softmax(hidden_layer @ weights[1] + biases[1])`.
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate values which are non-positive, and overflow does not occur.
        
        hidden_layer_value = ReLU(inputs @ weights[0] + biases[0])
        output_layer_value = softmax(hidden_layer_value @ weights[1] + biases[1])
        
        return hidden_layer_value, output_layer_value

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # The gradient used in SGD has now four parts, gradient of `weights[0]` and `weights[1]`
        # and gradient of `biases[0]` and `biases[1]`.
        #
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of $-log P(target | data)$, or you can compute
        # it step by step using the chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer,
        # - compute the derivative with respect to `weights[1]` and `biases[1]`,
        # - compute the derivative with respect to the hidden layer output,
        # - compute the derivative with respect to the hidden layer input,
        # - compute the derivative with respect to `weights[0]` and `biases[0]`.
        
        
        #weight1_gradient = 0
        #weight0_gradient = 0
        #bias1_gradient = 0
        #bias0_gradient = 0
        
        permutation_index = 0
        for i in range(len(train_data)// args.batch_size):
            
            weight1_gradient = 0
            weight0_gradient = 0
            bias1_gradient = 0
            bias0_gradient = 0
            
            for j in range(args.batch_size):
                index = permutation[permutation_index]
                
                train_targets = np.zeros(args.classes)
                train_targets [train_target[index]] = 1
                
                hidden_layer_value, output_layer_value = forward(train_data[index])
                
                derivative1 = output_layer_value - train_targets
                #derivative2 = np.outer(derivative1, hidden_layer_value)
                derivative2 = np.outer(hidden_layer_value, derivative1)
                
                bias1_gradient += derivative1
                weight1_gradient += derivative2
                
                derivative3 = weights[1] @ derivative1
                relu = hidden_layer_value > 0
                derivative4 = derivative3 * relu
                #derivative5 = np.outer(derivative4, train_data[index])
                derivative5 = np.outer(train_data[index], derivative4)
                
                bias0_gradient += derivative4
                weight0_gradient += derivative5
                
                permutation_index += 1
            
            weights[0] -= args.learning_rate * (weight0_gradient/ args.batch_size)
            weights[1] -= args.learning_rate * (weight1_gradient/ args.batch_size)
            biases [0] -= args.learning_rate * (bias0_gradient / args.batch_size)
            biases [1] -= args.learning_rate * (bias1_gradient / args.batch_size)
            



        # TODO: After the SGD epoch, measure the accuracy for both the
        # train test and the test set.
        train_targets = np.argmax(forward(train_data)[1], axis=1)
        test_targets =  np.argmax(forward(test_data)[1],axis=1)
        
        train_accuracy = sklearn.metrics.accuracy_score(train_target, train_targets)
        test_accuracy= sklearn.metrics.accuracy_score(test_target, test_targets)

        print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
            epoch + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases), [100 * train_accuracy, 100 * test_accuracy]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(args)
    print("Learned parameters:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:12]] + ["..."]) for ws in parameters), sep="\n")


    
# Rasto Nowak

# 6a81285c-247a-11ec-986f-f39926f24a9c

# Martin Oravec

# 1056cfa0-24fb-11ec-986f-f39926f24a9c