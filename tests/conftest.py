import os

import pytest
import pandas as pd


@pytest.fixture
def testpath():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def df_review():
    return pd.read_csv(testpath)
