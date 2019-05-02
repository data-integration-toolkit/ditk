import os
import sys
import pytest
import numpy as np

from datasets import Data

"""
datasets = [
            'midwest_survey',
            'employee_salaries',
            'medical_charge',
            'traffic_violations',
            'road_safety',
            'docs_payments',
            'beer_reviews',
            ]
"""
datasets = ['employee_salaries']


@pytest.mark.parametrize("dataset", datasets)
def test_datasets(dataset):
    data = Data(dataset).get_df()

    # There is one target variable
    assert len([col for col in data.col_action
                if data.col_action[col] == 'y']) == 1

    # There is only one 'se' variable
    assert len([col for col in data.col_action
                if data.col_action[col] == 'se']) == 1

    # Fetch only columns in data.col_action
    assert len(data.df.columns) == len(data.col_action)

    for name, action in data.col_action.items():
        assert action in ['y', 'se', 'num', 'ohe', 'ohe-1']
        if action == 'num':
            assert data.df[name].dtype in [np.dtype('int64'),
                                           np.dtype('float64')]
        # Missing values
        assert (data.df[name].isna().sum() == 0), \
            ("Error in: dataset '%s', column '%s'" % (dataset, name))
