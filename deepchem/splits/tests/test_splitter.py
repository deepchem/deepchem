"""
Tests for splitter objects.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar, Aneesh Pappu"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

from deepchem.splits import RandomSplitter
from deepchem.splits import ScaffoldSplitter
from deepchem.splits import StratifiedSplitter
from deepchem.datasets.tests import TestDatasetAPI
import pandas as pd
import math


class TestSplitters(TestDatasetAPI):
    """
    Test some basic splitters.
    """

    def test_singletask_random_split(self):
        """
        Test singletask RandomSplitter class.
        """
        solubility_dataset = self.load_solubility_data()
        random_splitter = RandomSplitter()
        train_data, valid_data, test_data = \
            random_splitter.train_valid_test_split(
                solubility_dataset,
                self.train_dir, self.valid_dir, self.test_dir,
                frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

    def test_singletask_scaffold_split(self):
        """
        Test singletask ScaffoldSplitter class.
        """
        solubility_dataset = self.load_solubility_data()
        scaffold_splitter = ScaffoldSplitter()
        train_data, valid_data, test_data = \
            scaffold_splitter.train_valid_test_split(
                solubility_dataset,
                self.train_dir, self.valid_dir, self.test_dir,
                frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

    def test_multitask_random_split(self):
        """
        Test multitask RandomSplitter class.
        """
        multitask_dataset = self.load_multitask_data()
        random_splitter = RandomSplitter()
        train_data, valid_data, test_data = \
            random_splitter.train_valid_test_split(
                multitask_dataset,
                self.train_dir, self.valid_dir, self.test_dir,
                frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

    def test_multitask_scaffold_split(self):
        """
        Test multitask ScaffoldSplitter class.
        """
        multitask_dataset = self.load_multitask_data()
        scaffold_splitter = ScaffoldSplitter()
        train_data, valid_data, test_data = \
            scaffold_splitter.train_valid_test_split(
                multitask_dataset,
                self.train_dir, self.valid_dir, self.test_dir,
                frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

    def test_stratified_multitask_split(self):
        """
        Test multitask StratifiedSplitter class
        """
        #ensure sparse dataset is actually sparse

        sparse_dataset = self.load_sparse_multitask_dataset()
        X, y, w, ids = sparse_dataset.to_numpy()

        ############################################ DEBUG

        print("y[11]")
        print(y[11])
        print("w[11]")
        print(w[11])
        ############################################ DEBUG
        assert 0 == 1

        """
        sparsity is determined by number of w weights that are 0 for a given task
        structure of w np array is such that each row corresponds to a sample -- e.g., analyze third column for third
        sparse task
        """
        y_np = y
        sparse_np = w #weight np array
        #debugging
        #print(sparse_np[:, 3])

        frac_train = 0.5
        cutoff = int(math.floor(frac_train * len(sparse_np)))
        sparse_np = sparse_np[:cutoff, :]
        y_np = y_np[:cutoff, :]

        #debugging
        #print(sparse_np[:, 3])

        sparse_df = pd.DataFrame(data = sparse_np)
        y_df = pd.DataFrame(data = y_np)
        #debugging
        #print(sparse_df.iloc[:, 3])

        total_rows = len(sparse_df.index)
        sparse_flag = False
        colIndex = 0
        for col in sparse_df:
            column = sparse_df[col] #get series representation
            zero_count = column.value_counts()[0]
            if zero_count == total_rows:
                print("good -- one column doesn't have results")
                print(colIndex)
                print("weight column")
                print(column)
                print("corresponding y column")
                print(y_df.iloc[:, colIndex])
                sparse_flag = True
                assert zero_count == total_rows
                break
        colIndex+=1
        if not sparse_flag:
            print("dataset isn't sparse")
            assert sparse_flag is True
        else:
            print("dataset is sparse")

        stratified_splitter = StratifiedSplitter()
        train_data, valid_data, test_data = \
            stratified_splitter.train_valid_test_split(
                sparse_dataset,
                self.train_dir, self.valid_dir, self.test_dir,
                frac_train=0.8, frac_valid=0.1, frac_test=0.1
            )

        datasets = [train_data, valid_data, test_data]
        datasetIndex = 0
        for dataset in datasets:
            np_list = dataset.to_numpy()
            y = np_list[1]
            # verify that each task in the train dataset has some hits
            y_df = pd.DataFrame(data=y)
            totalRows = len(y_df.index)
            for col in y_df:
                column = y_df[col]
                NaN_count = column.isnull().sum()
                if NaN_count == totalRows:
                    print("fail -- one column doesn't have results")
                    if datasetIndex == 0:
                        print("train_data failed")
                    elif datasetIndex == 1:
                        print("valid_data failed")
                    elif datasetIndex == 2:
                        print("test_data failed")
                    assert NaN_count != totalRows
            if datasetIndex == 0:
                print("train_data passed")
            elif datasetIndex == 1:
                print("valid_data passed")
            elif datasetIndex == 2:
                print("test_data passed")
            datasetIndex+=1
        print("end of stratified test")
        assert 1 == 1
