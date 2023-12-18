#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=False, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=1.0, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=44, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
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
    
    def __init__(self, max_depth, feature_subsampling):
        self._max_depth = max_depth
        self._feature_subsampling = feature_subsampling
        self._depth = 1

    def fit(self, data, targets):
        self._data = data
        self._targets = targets

        self._root = self._leafify(np.arange(len(self._data)), True) # node root has all instances
        self._recursive_approach(self._root, 0)
    
    def _recursive_approach(self, node, depth):
        if not self.can_split(node, depth):
            return
        
        criterion, feature, value, left, right = self._best_split(node)
        
        if criterion == 0:
            return
        
        node.split(feature, value, self._leafify(left), self._leafify(right, True))
        self._recursive_approach(node.left, depth + 1) # left first
        self._recursive_approach(node.right, depth + 1) # than right

    def _calculate_criterion(self, sorted_indices, feature, i):
        value = (self._data[sorted_indices[i], feature] + self._data[sorted_indices[i + 1], feature]) / 2
        left, right = sorted_indices[:i + 1], sorted_indices[i + 1:]
        return self.entropy(left) + self.entropy(right), value, left, right

    def _best_split(self, node): # criterion_difference, feature_index, best_value, left, right
        best_criterion = np.inf
        subsampled_features = self._feature_subsampling(self._data.shape[1])
        for feature in subsampled_features:
            sorted_indices = node.instances[np.argsort(self._data[node.instances, feature])] # TODO check
            for i in range(len(sorted_indices) - 1):
                if self._data[sorted_indices[i], feature] == self._data[sorted_indices[i + 1], feature]:
                    # same dividing value
                    continue
                criterion, value, left, right = self._calculate_criterion(sorted_indices, feature, i)
                if criterion < best_criterion:
                    best_criterion, best_feature, best_value, best_left, best_right = \
                        criterion, feature, value, left, right

        return best_criterion - self.entropy(node.instances), best_feature, best_value, best_left, best_right
    
    def _leafify(self, instances, increment = False):
        classes, counts = np.unique(self._targets[instances], return_counts=True)
        return self.Node(instances, classes[np.argmax(counts)])

    def can_split(self, node, depth): 
        # can split if depth is not max depth or depth is smaller than max_depth
        # and instances is more than min_to_split (or max_leaves in None)
        # and criterion is not zero        
        return (self._max_depth is None or depth < self._max_depth) and \
               self.entropy(node.instances) > 0
    
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

class RandomForest:
    def __init__(self, max_depth, trees_count, feature_subsampling, data_bootstrap):
        self.max_depth = max_depth
        self.feature_subsampling = feature_subsampling
        self.data_bootstrap = data_bootstrap
        self.trees_count = trees_count
        self.trees = []
        self.class_count = None

    def fit(self, data, target):
        self.class_count = np.max(target) + 1
        for _ in range(self.trees_count):
            data_indices = self.data_bootstrap(data)
            tree = DecisionTree(self.max_depth, self.feature_subsampling)
            tree.fit(data[data_indices], target[data_indices])
            self.trees.append(tree)

    def predict(self, data):
        predictions = np.zeros((len(data), self.class_count))
        for tree in self.trees:
            for index, prediction in enumerate(tree.predict(data)):
                predictions[index, prediction] += 1
        return np.argmax(predictions, axis=1)

def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Create random generators.
    generator_feature_subsampling = np.random.RandomState(args.seed)
    def subsample_features(number_of_features: int) -> np.ndarray:
        return np.sort(generator_feature_subsampling.choice(
            number_of_features, size=int(args.feature_subsampling * number_of_features), replace=False))

    generator_bootstrapping = np.random.RandomState(args.seed)
    def bootstrap_dataset(train_data: np.ndarray) -> np.ndarray:
        if args.bagging:
            return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)
        return np.arange(len(train_data))

    # TODO: Create a random forest on the training data.
    #
    # Use a simplified decision tree from the `decision_tree` assignment:
    # - use `entropy` as the criterion
    # - use `max_depth` constraint, to split a node only if:
    #   - its depth is less than `args.max_depth`
    #   - the criterion is not 0 (the corresponding instance targets are not the same)
    # When splitting nodes, proceed in the depth-first order, splitting all nodes
    # in the left subtree before the nodes in right subtree.
    #
    # Additionally, implement:
    # - feature subsampling: when searching for the best split, try only
    #   a subset of features. Notably, when splitting a node (i.e., when the
    #   splitting conditions [depth, criterion != 0] are satisfied), start by
    #   generating the subsampled features using
    #     subsample_features(number_of_features)
    #   returning the features that should be used during the best split search.
    #   The features are returned in ascending order, so when `feature_subsampling == 1`,
    #   the `np.arange(number_of_features)` is returned.
    #
    # - train a random forest consisting of `args.trees` decision trees
    #
    # - if `args.bagging` is set, before training each decision tree
    #   create a bootstrap sample of the training data by calling
    #     dataset_indices = bootstrap_dataset(train_data)
    #   and if `args.bagging` is not set, use the original training data.
    #
    # During prediction, use voting to find the most frequent class for a given
    # input, choosing the one with the smallest class number in case of a tie.

    random_forest = RandomForest(args.max_depth, args.trees, subsample_features, bootstrap_dataset)
    random_forest.fit(train_data, train_target)

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = sklearn.metrics.accuracy_score(train_target, random_forest.predict(train_data))
    test_accuracy = sklearn.metrics.accuracy_score(test_target, random_forest.predict(test_data))

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
