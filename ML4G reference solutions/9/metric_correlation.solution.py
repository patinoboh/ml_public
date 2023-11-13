#!/usr/bin/env python3
import argparse
import dataclasses

import numpy as np
import scipy.stats

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrap_samples", default=100, type=int, help="Bootstrap samples")
parser.add_argument("--data_size", default=1000, type=int, help="Data set size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--validate", default=False, action="store_true", help="Validate Pearson")


class ArtificialData:
    @dataclasses.dataclass
    class Sentence:
        """ Information about a single dataset sentence."""
        gold_edits: int  # Number of required edits to be performed.
        predicted_edits: int  # Number of edits predicted by a model.
        predicted_correct: int  # Number of correct edits predicted by a model.
        human_rating: int  # Human rating of the model prediction.

    def __init__(self, args: argparse.Namespace):
        generator = np.random.RandomState(args.seed)

        self.sentences = []
        for _ in range(args.data_size):
            gold = generator.poisson(2)
            correct = generator.randint(gold + 1)
            predicted = correct + generator.poisson(0.5)
            human_rating = max(0, int(100 - generator.uniform(5, 8) * (gold - correct)
                                      - generator.uniform(8, 13) * (predicted - correct)))
            self.sentences.append(self.Sentence(gold, predicted, correct, human_rating))


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Create the artificial data.
    data = ArtificialData(args)

    # Create `args.bootstrap_samples` bootstrapped samples of the dataset by
    # sampling sentences of the original dataset, and for each compute
    # - average of human ratings,
    # - TP, FP, FN counts of the predicted edits.
    human_ratings, predictions = [], []
    generator = np.random.RandomState(args.seed)
    for _ in range(args.bootstrap_samples):
        # Bootstrap sample of the dataset.
        sentences = generator.choice(data.sentences, size=len(data.sentences), replace=True)

        # TODO: Append the average of human ratings of `sentences` to `human_ratings`.
        human_ratings.append(np.mean([sentence.human_rating for sentence in sentences]))

        # TODO: Compute TP, FP, FN counts of predicted edits in `sentences`
        # and append them to `predictions`.
        predictions.append((
            sum(sentence.predicted_correct for sentence in sentences),
            sum(sentence.predicted_edits - sentence.predicted_correct for sentence in sentences),
            sum(sentence.gold_edits - sentence.predicted_correct for sentence in sentences),
        ))

    # Compute Pearson correlation between F_beta score and human ratings
    # for betas between 0 and 2.
    betas, correlations = [], []
    for beta in np.linspace(0, 2, 201):
        betas.append(beta)

        # TODO: For each bootstrap dataset, compute the F_beta score using
        # the counts in `predictions` and then manually compute the Pearson
        # correlation between the computed scores and `human_ratings`. Append
        # the result to `correlations`.
        f_scores = [(1 + beta**2) * tp / (tp + fp + beta**2 * (tp + fn)) for tp, fp, fn in predictions]

        hm = np.mean(human_ratings)
        fm = np.mean(f_scores)
        correlation = np.sum((human_ratings - hm) * (f_scores - fm)) / \
            (np.sqrt(np.sum((human_ratings - hm) ** 2)) * np.sqrt(np.sum((f_scores - fm) ** 2)))
        correlations.append(correlation)

        if args.validate:
            assert np.isclose(correlation, np.corrcoef(human_ratings, f_scores, rowvar=False)[0, 1])
            assert np.isclose(correlation, scipy.stats.pearsonr(human_ratings, f_scores)[0])

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(betas, correlations)
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"Pearson correlation of $F_\beta$-score and human ratings")
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    # TODO: Assign the highest correlation to `best_correlation` and
    # store corresponding beta to `best_beta`.
    best_index = np.argmax(correlations)
    best_beta, best_correlation = betas[best_index], correlations[best_index]
    # Make sure the maximum is unique up to 1e-6.
    assert all(i == best_index or correlations[i] + 1e-6 < best_correlation for i in range(len(correlations)))

    return best_beta, best_correlation


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_beta, best_correlation = main(args)

    print("Best correlation of {:.3f} was found for beta {:.2f}".format(
        best_correlation, best_beta))
