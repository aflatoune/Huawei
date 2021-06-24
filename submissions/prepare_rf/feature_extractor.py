import numpy as np
import datetime
import pandas as pd
import time
from extract import PrepareExtractor
from _clean import DataCleaner


class FeatureExtractor:
    def __init__(self):
        self.count = 0

    def transform(self, X):
        if len(X) != 29592 or len(X) != 50862 or len(X) != 8202:
            cleaner = DataCleaner(drop_olt_recv=True)
            X = cleaner.clean_data(X)
            prep = PrepareExtractor()
            X, y = prep.get_data(X, size_sample=-1,
                                resample={'unit': '6H', 'func': 'mean'},
                                fast=True, name=self.count)
        print(X.shape)
        self.count += 1
        return X
