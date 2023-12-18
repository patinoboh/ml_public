#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import heapq

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


class DecisionTree:
    class Node:
        def __init__(self, instances, prediction):
            self.is_leaf = True            
            self.instances = instances
            self.prediction = prediction

        def split(self, feature, value, left, right):
            self.is_leaf = False
            self.feature = feature
            self.value = value
            self.left = left
            self.right = right
    
    def predict_dato(self, dato):
        node = self._root
        while not node.is_leaf:
            node = node.left if dato[node.feature] <= node.value else node.right
        return node.prediction
    
    def predict(self, data):
        # data.shape = (n, d)                
        results = np.zeros(len(data), dtype=np.int32)
        for i, dato in enumerate(data):
            results[i] = self.predict_dato(dato)
        return results
    
    def __init__(self, criterion, max_depth, min_to_split, max_leaves):
        self._criterion = getattr(self, criterion)
        self._max_depth = max_depth
        self._min_to_split = min_to_split
        self._max_leaves = max_leaves
        self._leaves = 0
        self._depth = 1

    def fit(self, data, targets):
        self._data = data
        self._targets = targets

        self._root = self._leafify(np.arange(len(self._data))) # node root has all instances
        
        if self._max_leaves is None:
            self._recursive_approach(self._root, 0)
        else:
            self._adaptive_approach()
    
    def _recursive_approach(self, node, depth):
        if not self.can_split(node, depth):
            return
        
        criterion, feature, value, left, right = self._best_split(node)
        
        if criterion == 0:
            return
        
        node.split(feature, value, self._leafify(left), self._leafify(right))
        self._recursive_approach(node.left, depth + 1) # left first
        self._recursive_approach(node.right, depth + 1) # than right

    def _adaptive_approach(self):
    #   repeatably split a leaf where the
    #   constraints are valid and the overall criterion value ($c_left + c_right - c_node$)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).        
        def split_value(node, index_in_heap, depth): # criterion_difference, heap_index, depth, node, feature_index, ...
            best_split = self._best_split(node)
            return (best_split[0], index_in_heap, depth, node, *best_split[1:])
                
        # Function to handle the split of a node
        def handle_split(node, index_in_heap, depth):
            if self._can_split(node, depth):
                heapq.heappush(heap, split_value(node, index_in_heap, depth))
        
        # Initialize heap with root node
        heap = [split_value(self._root, 0, 0)]

        i = 0
        # Loop to create splits until maximum number of leaves is reached
        while heap and i < self._max_leaves - 1:
            # Pop node with best split from heap
            _, _, depth, node, feature, value, left, right = heapq.heappop(heap)

            # Perform the split
            node.split(feature, value, self._leaf(left), self._leaf(right))

            # Handle the split of the child nodes
            handle_split(node.left, 2 * i + 1, depth + 1)
            handle_split(node.right, 2 * i + 2, depth + 1)
            
            i += 1

    def _calculate_criterion(self, sorted_indices, feature, i):
        value = (self._data[sorted_indices[i], feature] + self._data[sorted_indices[i + 1], feature]) / 2
        left, right = sorted_indices[:i + 1], sorted_indices[i + 1:]
        return self._criterion(left) + self._criterion(right), value, left, right

    def _best_split(self, node): # criterion_difference, feature_index, best_value, left, right
        best_criterion = np.inf
        for feature in range(self._data.shape[1]):
            sorted_indices = node.instances[np.argsort(self._data[node.instances, feature])] # TODO check
            for i in range(len(sorted_indices) - 1):
                if self._data[sorted_indices[i], feature] == self._data[sorted_indices[i + 1], feature]:
                    # same dividing value
                    continue
                criterion, value, left, right = self._calculate_criterion(sorted_indices, feature, i)
                if criterion < best_criterion:
                    best_criterion, best_feature, best_value, best_left, best_right = \
                        criterion, feature, value, left, right

        return best_criterion - self._criterion(node.instances), best_feature, best_value, best_left, best_right
    
    def _leafify(self, instances):
        self._leaves += 1
        classes, counts = np.unique(self._targets[instances], return_counts=True)
        return self.Node(instances, classes[np.argmax(counts)])

    def can_split(self, node, depth):
        # can split if depth is not max depth or depth is smaller than max_depth
        # and instances is more than min_to_split (or max_leaves in None)
        # and criterion is not zero        
        return (self._max_depth is None or depth < self._max_depth) and \
               (self._max_leaves is None or self._max_leaves > self._leaves) and \
               len(node.instances) >= self._min_to_split and \
               self._criterion(node.instances) > 0
    
    def gini(self, instances):
        # TODO: Compute the Gini criterion of the given instances.
        # If the instances are empty, the criterion is 0.
        # Otherwise, the criterion is computed as described in the documentation.
        # return 0
        if len(instances) == 0:
            return 0
        # get unique classes
        classes = np.unique(self._targets[instances])
        # get counts of each class
        counts = np.array([np.sum(self._targets[instances] == c) for c in classes])
        # compute gini
        return np.sum(counts * (1 - counts / len(instances)))
    
    def entropy(self, instances):
        # TODO: Compute the entropy of the given instances.
        # If the instances are empty, the entropy is 0.
        # Otherwise, the entropy is computed as described in the documentation.
        # return 0
        if len(instances) == 0:
            return 0
        # get unique classes
        classes = np.unique(self._targets[instances])
        # get counts of each class
        counts = np.array([np.sum(self._targets[instances] == c) for c in classes])
        # compute entropy
        return -np.sum(counts * np.log(counts / len(instances)))


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Manually create a decision tree on the training data.
    #
    # - For each node, predict the most frequent class (and the one with
    #   the smallest number if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split decreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (e.g., for four instances
    #   with values 1, 7, 3, 3, the split points are 2 and 5).
    #
    # - When `args.max_leaves` is `None`, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not `None`), repeatably split a leaf where the
    #   constraints are valid and the overall criterion value ($c_left + c_right - c_node$)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    decisiton_tree = DecisionTree(args.criterion, args.max_depth, args.min_to_split, args.max_leaves)
    decisiton_tree.fit(train_data, train_target)

    # TODO: Finally, measure the training and testing accuracy.    
    train_accuracy = sklearn.metrics.accuracy_score(train_target, decisiton_tree.predict(train_data))
    test_accuracy = sklearn.metrics.accuracy_score(test_target, decisiton_tree.predict(test_data))

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))


# Patrik Brocek
# 5ccdc432-238f-11ec-986f-f39926f24a9c

# Rasto Nowak
# 6a81285c-247a-11ec-986f-f39926f24a9c

# Martin Oravec
# 1056cfa0-24fb-11ec-986f-f39926f24a9c
