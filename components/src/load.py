# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import argparse
import datetime
from pathlib import Path
import yaml
import pandas as pd
from mltable import load
from sklearn.model_selection import train_test_split


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--base_data", default="../../base-mltable-folder/creditcard.csv" ,type=str)
    parser.add_argument("--train_data", default="../../training-mltable-folder" ,type=str)
    parser.add_argument("--test_data", default="../../test-mltable-folder" ,type=str)
    parser.add_argument("--validation_data", default="../../validation-mltable-folder" ,type=str)
    # parse args
    args = parser.parse_args()
    print("args received ", args)
    # return args
    return args


def main(args):
    """
    loading and split
    """
    base_dataframe = pd.read_csv(args.base_data)


    # divide df into features and target
    x = base_dataframe.drop(columns='Class')
    y = base_dataframe['Class']

    # sklearn split
    seed = 10
    test_size = .2
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    seed = 10
    validation_size = .3
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size, random_state=seed)
    
    # remerge dfs
    train_dataframe = x_train.merge(y_train, left_index=True, right_index=True)
    test_dataframe = x_test.merge(y_test, left_index=True, right_index=True)
    validation_dataframe = x_validation.merge(y_validation, left_index=True, right_index=True)

    # write train data in output path
    train_data_path = os.path.join(
        args.train_data, "train_data.csv"
    )
    train_dataframe.to_csv(
        train_data_path,
        index=False,
        header=True,
    )

    # write test data in output path
    test_data_path = os.path.join(
    args.test_data, "test_data.csv"
    )
    test_dataframe.to_csv(
        test_data_path,
        index=False,
        header=True,
    )

    # write validation data in output path
    validation_data_path = os.path.join(
        args.validation_data, "validation_data.csv"
    )

    validation_dataframe.to_csv(
        validation_data_path,
        index=False,
        header=True,
    )

    # Write MLTable yaml file as well in output folder
    # Since in this example we are not doing any preprocessing, we are just copying same yaml file from input,change it if needed


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
