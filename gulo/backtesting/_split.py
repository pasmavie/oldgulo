import numpy as np
import pandas as pd
from typing import List, Union


class SequentialValidation:
    def __init__(
        self,
        train_window: int,
        test_window: int,
        split_column: str = None,
        split_candidates: List = None,
    ):
        """
        :param train_window: number of past periods, including the ones satisfying the split condition, to use for training. Pass -1 to fetch all the past history
        :param test_window: number of periods after the split condition ends to include in the test set.
        :param split_column: sortable X column to use for the search of train and test indices. If not passed, split will default to the input df index
        :param split_candidates: optional pre-defined split points
        """
        self.train_window = train_window
        self.test_window = test_window
        self.split_column = split_column
        self.split_candidates = split_candidates
        self.used_splits = list()

    def split(self, X: pd.DataFrame, y: pd.Series = None) -> (np.array, np.array):
        """
        :param X: DataFrame
        :param y: Series with target variable, it matters only to adapt to any code style
        :return: generator yielding the split condition + relative train and test indices (as numpy arrays)
        Usage:
        >>> X = pd.DataFrame(np.array(([2010, 0, 1], [2011, 2, 3], [2012, 4, 5], [2013, 6, 7], [2014, 8, 9], [2015, 10, 11])))
        >>> X.set_index(0, inplace=True)
        >>> y = pd.Series(np.array([2010, 2011, 2012, 2013, 2014, 2015]))
        >>> [(y[train], y[test]) for (train, test) in SequentialValidation(train_window=2, test_window=1).split(X, y)]
            [(array([2011, 2012]), array([2013])), (array([2012, 2013]), array([2014])), (array([2013, 2014]), array([2015]))]
        """
        split_column = X.index
        if self.split_column:
            split_column = X[self.split_column]
        split_candidates = list(
            np.sort(np.unique(self.split_candidates or split_column))
        )
        train_window = (
            self.train_window - 1
        )  # the observations selected by the split condition are included in the training set
        test_window = self.test_window
        splits = list(
            filter(
                lambda x: (x - train_window >= 0)
                & (x + test_window < len(split_candidates)),
                map(lambda y: split_candidates.index(y), split_candidates),
            )
        )
        train_window_starts = (
            list(split_candidates[i - train_window] for i in splits)
            if train_window > 0
            else (split_candidates[0] for i in splits)
        )
        test_window_ends = list(split_candidates[i + test_window] for i in splits)
        splits = list(split_candidates[i] for i in splits)
        for train_start, split, test_end in zip(
            train_window_starts, splits, test_window_ends
        ):
            self.used_splits.append(split)
            train_indices = np.where(
                (split_column >= train_start) & (split_column <= split)
            )[0]
            test_indices = np.where(
                (split_column > split) & (split_column <= test_end)
            )[0]
            yield (train_indices, test_indices)


class ExpandingValidation(SequentialValidation):
    """
    Expanding window (train, test) splits
    """

    def __init__(self, test_window=1, split_column=None, split_candidates=None):
        super(ExpandingValidation, self).__init__(
            train_window=-1,
            test_window=test_window,
            split_column=split_column,
            split_candidates=split_candidates,
        )


class RollingValidation(SequentialValidation):
    """
    Rolling window (train, test) splits
    """

    def __init__(self, split_column=None, split_candidates=None):
        super(RollingValidation, self).__init__(
            train_window=1,
            test_window=1,
            split_column=split_column,
            split_candidates=split_candidates,
        )
